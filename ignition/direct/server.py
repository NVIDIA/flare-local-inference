"""A minimal non-streaming OpenAI-compatible endpoint over Ignition's direct diagnostic execution.

HTTP handler threads render and tokenize chat requests with the checkpoint's
Transformers tokenizer, then wait while the main thread, which owns the CUDA
work, schedules, packs and executes steps one at a time. Replies split GLM-style
`<think>` reasoning from content. A request the model fails reports the device
error code as Helios does: HTTP 500 with "prepared model reported device error N".
`/metrics` exposes Helios's speculative round counters and the direct loop's own.

The scheduling clock counts each step as its graph time, and wall time only while
the loop waits for requests. Graph replay times every step's graph; with a graph
times file it also saves its estimates there, from which eager execution, whose
steps take Python's time rather than the graph's, schedules instead.
"""

import json
import math
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .executor import RankSteps
from .scheduler import AdmissionError, CompletionError, Layout, Request, SamplingParameters, Scheduler

_DEFAULT_OUTPUT_LIMIT = 32768
_SAVE_SECONDS = 10.


class _Submission:
    def __init__(self, request):
        self.request, self.done = request, threading.Event()


class _Frontend:
    """Chat rendering, tokenization and reply decoding with the checkpoint's own tokenizer and template."""

    def __init__(self, server_config):
        from transformers import PreTrainedTokenizerFast

        directory = Path(server_config["assets"]["tokenizer"]["path"]).parent
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
        self.eos = set(server_config["frontend"]["eos_token_ids"])

    def render(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise AdmissionError("messages must be a nonempty list")
        options = dict(body.get("chat_template_kwargs") or {})
        for name in ("tools", "reasoning_effort"):
            if body.get(name) is not None:
                options[name] = body[name]
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=False, **options)
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **options)
        return [int(token) for token in tokens], text

    def reply(self, tokens, rendered):
        """Split generated text into reasoning and content when the prompt opens a think block."""
        text = self.tokenizer.decode([t for t in tokens if t not in self.eos], skip_special_tokens=False)
        tail = rendered.rsplit("<|assistant|>", 1)[-1]
        if tail.rfind("<think>") > tail.rfind("</think>"):
            reasoning, separator, content = text.partition("</think>")
            if not separator:
                return "", reasoning
            return content.strip(), reasoning.strip()
        return text, None


def _sampling(body, defaults):
    def clamp(name, value, default):
        bounds = defaults.get("ranges", {}).get(name, {"min": default, "max": default})
        return min(max(value, bounds["min"]), bounds["max"])

    def number(name):
        default = float(defaults.get(name, SamplingParameters.__dataclass_fields__[name].default))
        value = body.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdmissionError(f"{name} must be a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise AdmissionError(f"{name} must be finite")
        return float(clamp(name, value, default))

    def integer(name, default):
        value = body.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdmissionError(f"{name} must be an integer")
        return value if name == "seed" else clamp(name, value, default)

    return SamplingParameters(
        seed=integer("seed", int(defaults.get("seed", 0))), temperature=number("temperature"),
        top_p=number("top_p"), min_p=number("min_p"), repetition_penalty=number("repetition_penalty"),
        presence_penalty=number("presence_penalty"), frequency_penalty=number("frequency_penalty"),
        top_k=integer("top_k", int(defaults.get("top_k", -1))))


class DirectServer:
    """Own the scheduler, the rank-zero step runner and the HTTP listener for one prepared model."""

    def __init__(self, model, peers, *, bind, eager, served_names, verbose, graph_times=None):
        self.model, self.eager, self.verbose = model, eager, verbose
        self.layout = Layout.of(model)
        self.scheduler = Scheduler(self.layout)
        self.graph_times = None if graph_times is None else Path(graph_times)
        self.recorded = load_graph_times(self.graph_times, self.layout) if eager and graph_times is not None else {}
        self.clock_ns = 0
        self.saved = time.monotonic()
        self.frontend = _Frontend(model.server_config)
        self.defaults = model.server_config.get("sampling_defaults", {})
        self.served_names = tuple(served_names)
        self.inbox = queue.Queue()
        self.identities = iter(range(1, 1 << 62))
        self.lock = threading.Lock()
        self.outcomes = {}
        self.failure = None
        self.started = time.time()
        self.steps = RankSteps(model, peers, eager=eager)
        host, _, port = bind.rpartition(":")
        listener = type("Listener", (ThreadingHTTPServer,), {
            "address_family": socket.AF_INET6 if host.startswith("[") else socket.AF_INET})
        self.http = listener((host.strip("[]"), int(port)), self._handler())
        self.http.daemon_threads = True
        self.address = f"http://{bind}"

    # HTTP side.

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, payload, content_type="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(503 if server.failure else 200, {"status": "failed" if server.failure else "ok"})
                elif self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [
                        {"id": name, "object": "model", "owned_by": "flare"} for name in server.served_names]})
                elif self.path == "/metrics":
                    self._send(200, server.metrics().encode(), "text/plain; version=0.0.4")
                else:
                    self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})
                    return
                status, payload = server.chat(body)
                self._send(status, payload)

        return Handler

    def chat(self, raw):
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise AdmissionError("the request body must be a JSON object")
            if body.get("stream"):
                raise AdmissionError("direct diagnostic serving answers non-streaming requests only")
            if body.get("n", 1) != 1:
                raise AdmissionError("direct diagnostic serving returns one choice")
            prompt, rendered = self.frontend.render(body)
            limit = body.get("max_completion_tokens", body.get("max_tokens"))
            if limit is None:
                limit = min(_DEFAULT_OUTPUT_LIMIT, self.layout.context - len(prompt))
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise AdmissionError("the output limit must be an integer")
            request = Request(next(self.identities), prompt, limit, _sampling(body, self.defaults),
                              ignore_eos=bool(body.get("ignore_eos", False)))
            self.scheduler.validate(request)
        except (AdmissionError, ValueError, TypeError) as error:
            return 400, {"error": {"message": str(error), "type": "invalid_request_error"}}
        if self.failure is not None:
            return 503, {"error": {"message": f"direct execution stopped: {self.failure}", "type": "unavailable"}}
        submission = _Submission(request)
        self.inbox.put(submission)
        submission.done.wait()
        request = submission.request
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(request.generated),
                 "total_tokens": len(prompt) + len(request.generated),
                 "prompt_tokens_details": {"cached_tokens": request.reused}}
        diagnostics = {"execution": "eager" if self.eager else "graph replay", "request": request.ident,
                       "speculative_rounds": request.rounds, "reused_tokens": request.reused}
        if request.error is not None:
            if isinstance(request.error, int):
                message = f"prepared model reported device error {request.error}"
                diagnostics["device_error"] = request.error
            else:
                message = str(request.error)
            return 500, {"error": {"message": message, "type": "execution_error", "code": diagnostics.get("device_error")},
                         "usage": usage, "flare_direct": diagnostics}
        content, reasoning = self.frontend.reply(request.generated, rendered)
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return 200, {"id": f"chatcmpl-direct-{request.ident}", "object": "chat.completion",
                     "created": int(time.time()), "model": body.get("model") or self.served_names[0],
                     "choices": [{"index": 0, "message": message, "finish_reason": request.finish_reason}],
                     "usage": usage, "flare_direct": diagnostics}

    def metrics(self):
        s = self.scheduler
        with self.lock:
            rounds, steps, errors = dict(s.rounds), dict(s.steps), dict(s.device_errors)
            outcomes, stats = dict(self.outcomes), dict(s.stats)
        lines = ["# TYPE flare_speculative_rounds_total counter"]
        lines += [f'flare_speculative_rounds_total{{batch="{b}",width="{w}"}} {n}' for (b, w), n in sorted(rounds.items())]
        lines.append("# TYPE flare_direct_steps_total counter")
        lines += [f'flare_direct_steps_total{{kind="{kind}"}} {n}' for kind, n in sorted(steps.items())]
        lines.append("# TYPE flare_direct_device_errors_total counter")
        lines += [f'flare_direct_device_errors_total{{code="{code}"}} {n}' for code, n in sorted(errors.items())]
        lines.append("# TYPE flare_direct_requests_total counter")
        lines += [f'flare_direct_requests_total{{outcome="{name}"}} {n}' for name, n in sorted(outcomes.items())]
        lines += [f"flare_direct_{name}_total {n}" for name, n in sorted(stats.items())]
        return "\n".join(lines) + "\n"

    # Execution side, on the thread that prepared the model.

    def _deliver(self, request):
        if request.error is None:
            outcome = "ok"
        elif isinstance(request.error, int):
            outcome = f"device_error_{request.error}"
        else:
            outcome = "failed"
        with self.lock:
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        error = "" if request.error is None else f" · {outcome if isinstance(request.error, int) else request.error}"
        print(f"request {request.ident}: prompt {len(request.prompt)} tokens (reused {request.reused}) · "
              f"{len(request.generated)} generated in {request.rounds} rounds · {request.finish_reason}{error}",
              file=sys.stderr, flush=True)
        self.pending.pop(request.ident).done.set()

    def _take(self, block):
        started = time.monotonic_ns()
        try:
            submission = self.inbox.get(timeout=.5) if block else self.inbox.get_nowait()
        except queue.Empty:
            return False
        finally:
            if block:
                # Waiting for requests is the only wall time the scheduling clock counts.
                self.clock_ns += time.monotonic_ns() - started
        self.scheduler.advance_to(self.clock_ns)
        while True:
            self.pending[submission.request.ident] = submission
            try:
                self.scheduler.submit(submission.request)
            except AdmissionError as error:
                submission.request.error = str(error)
                submission.request.finish_reason = "error"
                self._deliver(submission.request)
            try:
                submission = self.inbox.get_nowait()
            except queue.Empty:
                return True

    def serve(self):
        """Run steps until SIGINT or SIGTERM, which take effect between steps so every rank finishes the last one."""
        self.pending = {}
        thread = threading.Thread(target=self.http.serve_forever, name="flare-direct-http", daemon=True)
        thread.start()
        stopping = threading.Event()

        def stop(signum, frame):
            stopping.set()

        previous = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            while not stopping.is_set():
                self._take(block=self.scheduler.idle)
                with self.lock:
                    self.scheduler.advance_to(self.clock_ns)
                    step = self.scheduler.plan()
                    uploads = None if step is None else self.scheduler.pack(step)
                if step is None:
                    self._take(block=True)
                    continue
                measured = self.steps.run({"graph": step.graph, "uploads": uploads, "frame": step.frame,
                                           "sequence": step.sequence})
                graph_ns = self.recorded.get(step.graph) if self.eager else measured
                self.clock_ns += graph_ns or 0
                records = self.steps.records(step.frame)
                confidence = self.steps.confidence(step.graph) if (step.graph in self.scheduler.layout.paired_graphs
                    or step.kind == "packed_speculative") else None
                with self.lock:
                    self.scheduler.advance_to(self.clock_ns)
                    if confidence is not None and step.kind != "packed_speculative":
                        for lane, values in zip(step.lanes, confidence):
                            lane.request.acceptance.confidence(values)
                    retired = self.scheduler.complete(step, records, graph_ns, confidence=confidence)
                if self.verbose:
                    scheduled = f" (scheduled as {graph_ns / 1e6:.2f} ms)" if self.eager and graph_ns else ""
                    print(f"step {step.sequence} {step.graph} rows {len(step.lanes)} {measured / 1e6:.2f} ms"
                          f"{scheduled}", file=sys.stderr, flush=True)
                for request in retired:
                    self._deliver(request)
                if self.graph_times is not None and not self.eager and time.monotonic() - self.saved > _SAVE_SECONDS:
                    self._save_graph_times()
            return 0
        except (CompletionError, RuntimeError) as error:
            traceback.print_exc()
            self.failure = str(error)
            print(f"direct execution stopped: {error}", file=sys.stderr, flush=True)
            return 1
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            if self.graph_times is not None and not self.eager:
                self._save_graph_times()
            with self.lock:
                retired = self.scheduler.fail_all(self.failure or "direct execution stopped")
            for request in retired:
                self._deliver(request)
            while True:
                try:
                    submission = self.inbox.get_nowait()
                except queue.Empty:
                    break
                submission.request.error = "direct execution stopped"
                submission.done.set()
            self.http.shutdown()
            self.http.server_close()
            self.steps.stop()

    def _save_graph_times(self):
        """Write graph replay's current estimate of each graph's time, by graph name, replacing the file."""
        with self.lock:
            estimates = self.scheduler.policy.costs.estimates()
        names = {name: estimates[shape] for shape, name in self.layout.graphs.items() if shape in estimates}
        temporary = self.graph_times.with_name(self.graph_times.name + ".tmp")
        self.saved = time.monotonic()
        try:
            temporary.write_text(json.dumps({"graph_ns": dict(sorted(names.items()))}, indent=1) + "\n")
            os.replace(temporary, self.graph_times)
        except OSError as error:
            print(f"cannot save graph times to {self.graph_times}: {error}", file=sys.stderr, flush=True)


def load_graph_times(path, layout):
    """Graph times a graph replay run saved, for this model's graphs; unknown names are ignored."""
    try:
        saved = json.loads(Path(path).read_text())["graph_ns"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AdmissionError(f"cannot read graph times from {path}: {error}") from error
    names = set(layout.graphs.values())
    return {name: int(nanos) for name, nanos in saved.items()
            if name in names and isinstance(nanos, int) and nanos > 0}


def serve_direct(prepared, *, bind, eager, served_names, verbose=False, graph_times=None):
    """Serve a prepared model, or a rank group's rank zero, through Ignition's direct step loop."""
    model = getattr(prepared, "model", prepared)
    with prepared:
        ranks = [{key: value for key, value in rank.items() if key not in ("tuning", "breakdown")}
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,))]
        print(json.dumps({"memory": ranks}, default=str), file=sys.stderr, flush=True)
        peers = prepared.start_direct(eager=eager) if prepared is not model else ()
        server = DirectServer(model, peers, bind=bind, eager=eager, served_names=served_names, verbose=verbose,
                              graph_times=graph_times)
        layout = server.layout
        widths = ("exactly " + str(layout.fixed_width - 1) if layout.fixed_width else
                  "up to " + str(layout.max_width - 1)) + " draft tokens" if layout.speculative else "speculation off"
        print(f"Direct diagnostic execution ({'eager forwards' if eager else 'graph replay'}; not production serving): "
              f"KV {layout.pages:,} pages · {layout.pages * layout.page_tokens:,} tokens"
              f" · {model.memory_report.get('state_rows')} state rows ({layout.snapshots} for snapshots)"
              f" · concurrency {layout.max_active} · {widths}", flush=True)
        if eager:
            print(f"Eager steps are scheduled as {len(server.recorded)} graph times from {graph_times}" if graph_times
                  else "Eager steps have no graph times (--graph-times): every decode round carries every decoding"
                       " request at the widest verification, as Helios chooses among unmeasured graphs", flush=True)
        elif graph_times is not None:
            print(f"Graph times are saved to {graph_times}", flush=True)
        print(f"Listening on {server.address}", flush=True)
        return server.serve()
