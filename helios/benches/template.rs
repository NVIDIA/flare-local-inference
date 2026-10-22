use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{Value as Json, json};

#[path = "../src/frontend/jinja/mod.rs"]
mod jinja;
#[path = "../src/frontend/json.rs"]
mod json;

struct Allocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Relaxed) {
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        pointer
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
            let live = if size >= layout.size() {
                LIVE.fetch_add(size - layout.size(), Relaxed) + size - layout.size()
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed) - (layout.size() - size)
            };
            PEAK.fetch_max(live, Relaxed);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn environment(source: &str) -> Environment<'static> {
    let mut environment = Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_filter("tojson", json::tojson as fn(_, _) -> _);
    environment.add_test("iterable", |v: &Value| !v.is_none() && v.try_iter().is_ok());
    environment.add_function("raise_exception", |v: Value| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, v.to_string()))
    });
    environment
        .add_template_owned("chat", source.to_owned())
        .unwrap();
    environment
}

fn measure(name: &str, mut run: impl FnMut()) {
    for _ in 0..10 {
        run();
    }
    let mut times = Vec::with_capacity(101);
    for _ in 0..101 {
        let start = Instant::now();
        run();
        times.push(start.elapsed().as_nanos());
    }
    times.sort_unstable();
    for counter in [&ALLOCATIONS, &BYTES, &LIVE, &PEAK] {
        counter.store(0, Relaxed);
    }
    COUNTING.store(true, Relaxed);
    run();
    COUNTING.store(false, Relaxed);
    println!(
        "{name},median_ns={},p95_ns={},allocations={},allocated_bytes={},peak_live_bytes={}",
        times[50],
        times[95],
        ALLOCATIONS.load(Relaxed),
        BYTES.load(Relaxed),
        PEAK.load(Relaxed)
    );
}

fn footprint<T>(name: &str, build: impl FnOnce() -> T) {
    for counter in [&ALLOCATIONS, &BYTES, &LIVE, &PEAK] {
        counter.store(0, Relaxed);
    }
    COUNTING.store(true, Relaxed);
    let compiled = build();
    COUNTING.store(false, Relaxed);
    println!("{name},compiled_live_bytes={}", LIVE.load(Relaxed));
    drop(compiled);
}

fn concurrent(name: &str, iterations: usize, run: impl Fn() + Sync) {
    let barrier = std::sync::Barrier::new(5);
    let elapsed = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let run = &run;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..iterations {
                        run();
                    }
                })
            })
            .collect();
        let start = Instant::now();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        start.elapsed()
    });
    println!(
        "{name},threads=4,requests={},requests_per_second={:.1}",
        iterations * 4,
        iterations as f64 * 4.0 / elapsed.as_secs_f64()
    );
}

fn contexts() -> Vec<(&'static str, Json)> {
    let user = json!({"role":"user", "content":"Hello, café 世界 👋.\nReturn a concise answer."});
    let assistant = json!({"role":"assistant", "content":"An answer.", "reasoning_content":"Consider the inputs."});
    let base = json!({"messages":[user], "tools":null, "documents":null,
        "add_generation_prompt":true, "bos_token":"<s>", "eos_token":"</s>", "reasoning_effort":"low"});
    let mut cases = vec![("short", base.clone())];
    let mut history = base.clone();
    history["messages"] = Json::Array(
        (0..1000)
            .map(|i| {
                if i % 2 == 0 {
                    user.clone()
                } else {
                    assistant.clone()
                }
            })
            .collect(),
    );
    cases.push(("history_1000", history));
    let mut large = base.clone();
    large["messages"][0]["content"] = "Code and prose café 世界.\n".repeat(50_000).into();
    cases.push(("large_message", large));
    let mut tools = base.clone();
    tools["tools"] = json!([{"type":"function", "function":{"name":"lookup", "parameters":{"type":"object", "properties":{"query":{"type":"string"}}, "required":["query"]}}}]);
    tools["messages"] = json!([user, {"role":"assistant", "content":"", "tool_calls":[{"id":"a", "function":{"name":"lookup", "arguments":{"query":"café"}}}]}, {"role":"tool", "tool_call_id":"a", "name":"lookup", "content":"Result"}]);
    cases.push(("tools", tools));
    let mut images = base;
    images["messages"][0]["content"] = json!([{"type":"text", "text":"Describe "}, {"type":"image"}, {"type":"text", "text":" precisely."}]);
    cases.push(("images", images));
    cases
}

fn main() {
    let check_only = std::env::args().any(|a| a == "--check");
    let owned_only = std::env::args().any(|a| a == "--owned-only");
    for argument in std::env::args().skip(1).filter(|a| !a.starts_with("--")) {
        let (name, path) = argument
            .split_once('=')
            .expect("arguments are NAME=TEMPLATE_PATH");
        let source = std::fs::read_to_string(path)
            .unwrap()
            .replace("{%- generation %}", "{%- if true %}")
            .replace("{%- endgeneration %}", "{%- endif %}");
        let compiled = jinja::CompiledTemplate::compile(&source).unwrap();
        let mut unmarked = source.clone();
        for span in compiled.annotation_spans().iter().rev() {
            unmarked.replace_range(span.clone(), "");
        }
        if !check_only && !owned_only {
            measure(&format!("minijinja,{name},compile"), || {
                black_box(environment(&unmarked));
            });
        }
        let environment = environment(&unmarked);
        let template = environment.get_template("chat").unwrap();
        if !check_only {
            measure(&format!("owned,{name},compile"), || {
                black_box(jinja::CompiledTemplate::compile(&source).unwrap());
            });
        }
        println!("owned,{name},bytecode_bytes={}", compiled.bytecode_bytes());
        if !check_only {
            footprint(&format!("owned,{name}"), || {
                jinja::CompiledTemplate::compile(&source).unwrap()
            });
            footprint(&format!("minijinja,{name}"), || {
                self::environment(&unmarked)
            });
        }
        for (case, context) in contexts() {
            let start = Instant::now();
            let expected = template.render(&context).unwrap();
            let iterations =
                (50_000_000 / start.elapsed().as_nanos().max(1)).clamp(1, 5000) as usize;
            let actual = compiled.render(context.as_object().unwrap(), &[]).unwrap();
            assert_eq!(actual.text, expected, "{name}/{case}");
            if check_only {
                println!(
                    "{name}/{case}: output matches, {} marks",
                    actual.marks.len()
                );
                continue;
            }
            if !owned_only {
                measure(&format!("minijinja,{name},{case}"), || {
                    let mut output = Vec::new();
                    template
                        .render_captured_to(black_box(&context), &mut output)
                        .unwrap();
                    black_box(output);
                });
            }
            measure(&format!("owned,{name},{case}"), || {
                black_box(
                    compiled
                        .render(black_box(context.as_object().unwrap()), &[])
                        .unwrap(),
                );
            });
            concurrent(&format!("minijinja,{name},{case}"), iterations, || {
                let mut output = Vec::new();
                template.render_captured_to(&context, &mut output).unwrap();
                black_box(output);
            });
            concurrent(&format!("owned,{name},{case}"), iterations, || {
                black_box(compiled.render(context.as_object().unwrap(), &[]).unwrap());
            });
        }
    }
}
