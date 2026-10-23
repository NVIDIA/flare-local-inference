use super::super::metrics::scheduler::{
    Dispatch, GraphLabel, GraphSample, RequestDetail, SchedulerDetail, SchedulerMetrics,
    TRACE_CAPACITY,
};
use super::*;
use crate::{
    execution::GraphKind,
    scheduler::{RequestHandle, RequestPhase},
};

const INTERVAL_NS: u64 = 2_000_000_000;
const HISTORY_NS: u64 = 10_000_000_000;
const HISTORY_CAPACITY: usize = 16_384;
const REQUEST_ROWS: usize = 128;

pub(super) struct SchedulerView {
    now_ns: u64,
    incoming: VecDeque<GraphSample>,
    samples: VecDeque<GraphSample>,
    detail: SchedulerDetail,
    dispatch: Option<Dispatch>,
    truncated_at_ns: Option<u64>,
    offset: usize,
}

impl Default for SchedulerView {
    fn default() -> Self {
        Self {
            now_ns: 0,
            incoming: VecDeque::with_capacity(TRACE_CAPACITY),
            samples: VecDeque::with_capacity(HISTORY_CAPACITY),
            detail: SchedulerDetail::default(),
            dispatch: None,
            truncated_at_ns: None,
            offset: 0,
        }
    }
}

#[derive(Default)]
struct GraphTotal {
    graph: usize,
    calls: usize,
    timed: usize,
    gpu_ns: u64,
    max_ns: u64,
    lanes: usize,
}

impl SchedulerView {
    pub fn sample(&mut self, metrics: &SchedulerMetrics, now_ns: u64) {
        self.now_ns = now_ns;
        (self.detail, self.dispatch) = metrics.drain(&mut self.incoming);
        for sample in self.incoming.drain(..) {
            if self.samples.len() == HISTORY_CAPACITY {
                self.truncated_at_ns = self.samples.pop_front().map(|s| s.at_ns);
            }
            self.samples.push_back(sample);
        }
        while self
            .samples
            .front()
            .is_some_and(|s| now_ns.saturating_sub(s.at_ns) >= HISTORY_NS)
        {
            self.samples.pop_front();
        }
        self.offset = self.offset.min(self.requests().len().saturating_sub(1));
    }

    pub fn scroll(&mut self, delta: isize) {
        self.offset = self
            .offset
            .saturating_add_signed(delta)
            .min(self.requests().len().saturating_sub(1));
    }

    fn totals(&self, graphs: usize) -> Vec<GraphTotal> {
        let mut totals: Vec<_> = (0..graphs)
            .map(|graph| GraphTotal {
                graph,
                ..Default::default()
            })
            .collect();
        for sample in self
            .samples
            .iter()
            .filter(|s| self.now_ns.saturating_sub(s.at_ns) < INTERVAL_NS)
        {
            let Some(total) = totals.get_mut(sample.graph) else {
                continue;
            };
            total.calls += 1;
            total.lanes += sample.generations.iter().filter(|g| **g != 0).count();
            if let Some(ns) = sample.gpu_ns {
                total.timed += 1;
                total.gpu_ns = total.gpu_ns.saturating_add(ns);
                total.max_ns = total.max_ns.max(ns);
            }
        }
        totals.retain(|t| t.calls > 0);
        totals.sort_by_key(|t| (std::cmp::Reverse(t.gpu_ns), t.graph));
        totals
    }

    fn requests(&self) -> Vec<RequestHandle> {
        let mut rows: Vec<_> = self
            .detail
            .requests
            .iter()
            .flatten()
            .map(|r| r.handle)
            .collect();
        for sample in self.samples.iter().rev() {
            for (slot, generation) in sample.generations.iter().copied().enumerate() {
                if generation == 0 {
                    continue;
                }
                let handle = RequestHandle {
                    slot: slot as u32,
                    generation,
                };
                if !rows.contains(&handle) {
                    rows.push(handle);
                }
                if rows.len() == REQUEST_ROWS {
                    return rows;
                }
            }
        }
        rows
    }

    fn request_detail(&self, handle: RequestHandle) -> Option<RequestDetail> {
        self.detail.requests[handle.slot as usize].filter(|r| r.handle == handle)
    }

    fn activity(
        &self,
        handle: RequestHandle,
        width: usize,
        catalog: &[GraphLabel],
    ) -> Vec<Span<'static>> {
        let mut bins = vec![None; width];
        for sample in &self.samples {
            if sample.generations[handle.slot as usize] != handle.generation {
                continue;
            }
            let age = self.now_ns.saturating_sub(sample.at_ns);
            if age >= HISTORY_NS || width == 0 {
                continue;
            }
            let column = width - 1 - (age as u128 * width as u128 / HISTORY_NS as u128) as usize;
            if let Some(graph) = catalog.get(sample.graph) {
                let model_work = |kind| {
                    matches!(
                        kind,
                        GraphKind::Prefill
                            | GraphKind::Decode
                            | GraphKind::Speculative
                            | GraphKind::Vision
                    )
                };
                if model_work(graph.kind) || !bins[column].is_some_and(model_work) {
                    bins[column] = Some(graph.kind);
                }
            }
        }
        bins.into_iter()
            .map(|kind| match kind {
                Some(kind) => {
                    let (symbol, color) = kind_style(kind);
                    Span::styled(symbol, Style::default().fg(color))
                }
                None => Span::styled("·", Style::default().fg(Color::Rgb(50, 64, 80))),
            })
            .collect()
    }

    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        snapshot: Snapshot,
        metrics: &SchedulerMetrics,
    ) {
        let catalog = metrics.catalog.get().map_or(&[][..], Vec::as_slice);
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Percentage(42),
            Constraint::Min(4),
            Constraint::Length(2),
        ])
        .split(area);
        let last = self
            .dispatch
            .and_then(|d| catalog.get(d.graph).map(|g| (d, g)));
        let last_line = last.map_or_else(
            || " Waiting for graph submissions".to_owned(),
            |(d, g)| {
                let estimate = if d.estimate_ns == 0 {
                    "? (learning)".to_owned()
                } else {
                    format!("{:.2} ms", ms(d.estimate_ns))
                };
                format!(
                    " Last: {} · {} requests · estimate {}{} · {:.1}s ago",
                    g.name,
                    d.requests,
                    estimate,
                    if d.feedback { " · feedback" } else { "" },
                    self.now_ns.saturating_sub(d.at_ns) as f64 / 1e9
                )
            },
        );
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(format!(
                    " ACTIVE {}   QUEUED {}   IN FLIGHT {}   state age {:.1}s",
                    snapshot.active,
                    snapshot.pending,
                    self.detail.outstanding,
                    self.now_ns.saturating_sub(self.detail.at_ns) as f64 / 1e9
                )),
                Line::from(last_line).style(Style::default().fg(MUTED)),
                Line::from(self.completion_line(catalog)).style(Style::default().fg(MUTED)),
            ]),
            rows[0],
        );
        self.render_graphs(frame, rows[1], catalog);
        self.render_requests(frame, rows[2], catalog);
        let dropped = metrics.dropped.load(Ordering::Relaxed);
        let truncated = self
            .truncated_at_ns
            .is_some_and(|at| self.now_ns.saturating_sub(at) < HISTORY_NS);
        frame.render_widget(Paragraph::new(vec![
            Line::from(" GPU time: max rank interval. Share is of observed graph time, not utilization."),
            Line::from(format!(" ↑ ↓ requests   P prefill  D decode  S speculative  I/R/U/C/V setup   dropped {dropped}{}",
                if truncated { " · history truncated" } else { "" })),
        ]).style(Style::default().fg(MUTED)), rows[3]);
    }

    fn completion_line(&self, catalog: &[GraphLabel]) -> String {
        use crate::scheduler::{CostFreshness, TimingOrigin};
        let Some(sample) = self.samples.back() else {
            return " Waiting for graph completion timings".to_owned();
        };
        let name = catalog.get(sample.graph).map_or("?", |graph| graph.name.as_str());
        let status = match sample.selection.freshness {
            CostFreshness::Unknown => "unknown",
            CostFreshness::Fresh => "fresh",
            CostFreshness::Stale => "stale",
        };
        let origin = match sample.selection.origin {
            TimingOrigin::Ordinary => "service".to_owned(),
            TimingOrigin::Discovery => "discovery".to_owned(),
            TimingOrigin::Acceptance => sample.selection.refresh_episode
                .map_or_else(|| "acceptance".to_owned(), |id| format!("acceptance #{id}")),
            TimingOrigin::Refresh => sample.selection.refresh_episode
                .map_or_else(|| "refresh".to_owned(), |id| format!("refresh #{id}")),
        };
        let estimate = if sample.selection.retained_ns == 0 {
            "?".to_owned()
        } else {
            format!("{:.2}", ms(sample.selection.retained_ns))
        };
        let gpu = sample.gpu_ns.map_or_else(|| "?".to_owned(), |ns| format!("{:.2}", ms(ns)));
        let age = sample.selection.observed_at_ns.map_or_else(|| "?".to_owned(), |at| {
            format!("{:.1}s", sample.selection.selected_at_ns.saturating_sub(at) as f64 / 1e9)
        });
        format!(
            " Done #{} {name} · {origin}/{status} · est {estimate} GPU {gpu} wall {:.2} ms{} · age {age} · credit {:.2}→{:.2} ms",
            sample.fence.0, ms(sample.wall_ns), if sample.queued { " queued" } else { "" },
            sample.selection.credit_ns / 1e6, sample.credit_after_ns / 1e6,
        )
    }

    fn render_graphs(&self, frame: &mut Frame, area: Rect, catalog: &[GraphLabel]) {
        let totals = self.totals(catalog.len());
        let untimed: usize = totals.iter().map(|t| t.calls - t.timed).sum();
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(format!(" GRAPH COMPLETIONS · last 2s · {untimed} untimed "))
            .border_style(Style::default().fg(MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let total_ns: u64 = totals.iter().map(|t| t.gpu_ns).sum();
        let wide = inner.width >= 90;
        let mut lines = vec![
            Line::from(if wide {
                " Graph shape             Calls   Req/call   Mean ms    Max ms   GPU time share"
            } else {
                " Graph shape             Calls  Mean ms   GPU time share"
            })
            .style(Style::default().fg(MUTED)),
        ];
        if totals.is_empty() {
            lines.push(Line::from(" No graph completions in this interval."));
        }
        let available = inner.height.saturating_sub(1) as usize;
        let shown = if totals.len() > available {
            available.saturating_sub(1)
        } else {
            available
        };
        for t in totals.iter().take(shown) {
            let graph = &catalog[t.graph];
            let share = if total_ns == 0 {
                0.0
            } else {
                t.gpu_ns as f64 / total_ns as f64
            };
            let mean = if t.timed == 0 {
                "—".to_owned()
            } else {
                format!("{:.2}", ms(t.gpu_ns) / t.timed as f64)
            };
            let (_, color) = kind_style(graph.kind);
            let label = shape(graph);
            let max = if t.timed == 0 {
                "—".to_owned()
            } else {
                format!("{:.2}", ms(t.max_ns))
            };
            let prefix = if wide {
                format!(
                    " {label:<23} {:>5}  {:>8.2}  {mean:>8}  {max:>8}   ",
                    t.calls,
                    t.lanes as f64 / t.calls as f64,
                )
            } else {
                format!(" {label:<23} {:>5} {mean:>8}   ", t.calls)
            };
            let bar_width = (inner.width as usize)
                .saturating_sub(prefix.chars().count() + 7)
                .min(24);
            let filled = (share * bar_width as f64).round() as usize;
            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(color)),
                Span::styled("█".repeat(filled), Style::default().fg(color)),
                Span::styled(
                    "░".repeat(bar_width - filled),
                    Style::default().fg(Color::Rgb(50, 64, 80)),
                ),
                Span::raw(if total_ns == 0 {
                    "     —".to_owned()
                } else {
                    format!(" {:>5.1}%", share * 100.0)
                }),
            ]));
        }
        if totals.len() > shown && available > 0 {
            let omitted = &totals[shown..];
            lines.push(
                Line::from(format!(
                    " +{} other shapes · {} calls · {:.2} ms GPU",
                    omitted.len(),
                    omitted.iter().map(|t| t.calls).sum::<usize>(),
                    ms(omitted.iter().map(|t| t.gpu_ns).sum())
                ))
                .style(Style::default().fg(MUTED)),
            );
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn render_requests(&self, frame: &mut Frame, area: Rect, catalog: &[GraphLabel]) {
        let requests = self.requests();
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(format!(
                " REQUEST COMPLETIONS · last 10s · {} rows{} ",
                requests.len(),
                if requests.len() == REQUEST_ROWS {
                    " (limit)"
                } else {
                    ""
                }
            ))
            .border_style(Style::default().fg(MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let wide = inner.width >= 100;
        let header = if wide {
            format!(
                " {:<14} {:<10} {:>11} {:>6} {:>7} {:>7} {:>7} ",
                "Request", "Phase", "Prompt", "Output", "Gap ms", "Out ms", "Bursts"
            )
        } else {
            format!(" {:<14} {:<10} {:>6} ", "Request", "Phase", "Gap ms")
        };
        let prefix_width = header.chars().count();
        let width = (inner.width as usize).saturating_sub(prefix_width);
        let mut lines = vec![
            Line::from(format!(
                "{header}−10s{}now",
                " ".repeat(width.saturating_sub(7))
            ))
            .style(Style::default().fg(MUTED)),
        ];
        for handle in requests
            .iter()
            .skip(self.offset)
            .take(inner.height.saturating_sub(1) as usize)
        {
            let label = format!("{}.{}", handle.slot, handle.generation);
            let detail = self.request_detail(*handle);
            let prefix = match detail {
                Some(r) => {
                    let phase = phase(r.phase);
                    let gap = ms(self.detail.at_ns.saturating_sub(r.last_progress_ns));
                    if wide {
                        let output_gap = r.last_output_ns.map_or_else(
                            || "—".to_owned(),
                            |at| format!("{:.0}", ms(self.detail.at_ns.saturating_sub(at))),
                        );
                        format!(
                            " {label:<14} {phase:<10} {:>11} {:>6} {gap:>7.0} {output_gap:>7} {:>7} ",
                            format!("{}/{}", r.consumed.min(r.prompt as u64), r.prompt),
                            r.generated,
                            r.published_deliveries
                        )
                    } else {
                        format!(" {label:<14} {phase:<10} {gap:>6.0} ")
                    }
                }
                None => format!(
                    " {:<width$}",
                    format!("{label:<14} retired"),
                    width = prefix_width.saturating_sub(1)
                ),
            };
            let mut spans = vec![Span::styled(
                prefix,
                Style::default().fg(if detail.is_some() { TEXT } else { MUTED }),
            )];
            spans.extend(self.activity(*handle, width, catalog));
            lines.push(Line::from(spans));
        }
        if requests.is_empty() {
            lines.push(Line::from(" No active or recently served requests."));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn shape(graph: &GraphLabel) -> String {
    let name = match graph.kind {
        GraphKind::ScoredPrefill => "scored prefill",
        GraphKind::Prefill => "prefill",
        GraphKind::Decode => "decode",
        GraphKind::Speculative => "speculative",
        GraphKind::Proposal => "proposal",
        GraphKind::Verification => "verification",
        GraphKind::PackedSpeculative => "packed speculative",
        GraphKind::Mixed => "mixed",
        GraphKind::Init => "init",
        GraphKind::Restore => "restore",
        GraphKind::Resume => "resume",
        GraphKind::Snapshot => "snapshot",
        GraphKind::Vision => "vision",
    };
    if graph.kind == GraphKind::Mixed {
        format!(
            "{name} b{} p{} w{}",
            graph.batch, graph.prefill, graph.tokens
        )
    } else if graph.packed != 0 {
        format!("{name} b{} w{} rows{}", graph.batch, graph.tokens, graph.packed)
    } else if graph.kind == GraphKind::Speculative {
        format!(
            "{name} b{} d{}",
            graph.batch,
            graph.tokens.saturating_sub(1)
        )
    } else {
        format!("{name} b{} t{}", graph.batch, graph.tokens)
    }
}

fn kind_style(kind: GraphKind) -> (&'static str, Color) {
    match kind {
        GraphKind::Prefill | GraphKind::ScoredPrefill => ("P", CYAN),
        GraphKind::Decode => ("D", GREEN),
        GraphKind::Speculative => ("S", PURPLE),
        GraphKind::Proposal => ("Q", PURPLE),
        GraphKind::Verification => ("T", PURPLE),
        GraphKind::PackedSpeculative => ("S", PURPLE),
        GraphKind::Mixed => ("M", CYAN),
        GraphKind::Init => ("I", MUTED),
        GraphKind::Restore => ("R", GOLD),
        GraphKind::Resume => ("U", GOLD),
        GraphKind::Snapshot => ("C", MUTED),
        GraphKind::Vision => ("V", GOLD),
    }
}

fn phase(phase: RequestPhase) -> &'static str {
    match phase {
        RequestPhase::Initialize => "init",
        RequestPhase::Inbound => "inbound",
        RequestPhase::Restore => "restore",
        RequestPhase::Vision => "vision",
        RequestPhase::Prefill => "prefill",
        RequestPhase::Resume => "resume",
        RequestPhase::Decode => "decode",
        RequestPhase::Snapshot => "snapshot",
        RequestPhase::Finishing => "finishing",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn catalog() -> Vec<GraphLabel> {
        vec![
            GraphLabel {
                name: "prefill_1_1024".into(),
                kind: GraphKind::Prefill,
                batch: 1,
                tokens: 1024,
                prefill: 0,
                packed: 0,
            },
            GraphLabel {
                name: "speculative_4_6".into(),
                kind: GraphKind::Speculative,
                batch: 4,
                tokens: 6,
                prefill: 0,
                packed: 0,
            },
            GraphLabel {
                name: "snapshot_1".into(),
                kind: GraphKind::Snapshot,
                batch: 1,
                tokens: 1,
                prefill: 0,
                packed: 0,
            },
        ]
    }

    fn sample(at_ns: u64, graph: usize, generation: u64, gpu_ns: Option<u64>) -> GraphSample {
        let mut generations = [0; crate::execution::REQUEST_LANES];
        generations[0] = generation;
        GraphSample {
            at_ns,
            fence: crate::cache::Fence(at_ns),
            graph,
            gpu_ns,
            wall_ns: gpu_ns.unwrap_or(0),
            queued: false,
            selection: Default::default(),
            credit_after_ns: 0.0,
            generations,
        }
    }

    fn fixture() -> (SchedulerView, Metrics, History) {
        let metrics = Metrics::default();
        metrics.scheduler_trace.catalog.set(catalog()).ok().unwrap();
        let mut view = SchedulerView {
            now_ns: 10_000_000_000,
            ..Default::default()
        };
        view.samples.extend([
            sample(8_000_000_000, 0, 1, Some(99_000_000)),
            sample(9_000_000_000, 0, 1, Some(20_000_000)),
            sample(9_500_000_000, 1, 2, Some(5_000_000)),
            sample(9_600_000_000, 1, 2, Some(5_000_000)),
            sample(9_700_000_000, 1, 2, None),
            sample(9_800_000_000, 2, 2, Some(1_000_000)),
        ]);
        view.detail.at_ns = view.now_ns;
        view.detail.requests[0] = Some(RequestDetail {
            handle: RequestHandle {
                slot: 0,
                generation: 2,
            },
            phase: RequestPhase::Decode,
            consumed: 1027,
            prompt: 1024,
            generated: 3,
            last_progress_ns: 9_990_000_000,
            last_output_ns: Some(9_985_000_000),
            published_deliveries: 2,
        });
        view.dispatch = Some(Dispatch {
            at_ns: view.now_ns,
            graph: 1,
            requests: 1,
            estimate_ns: 0,
            feedback: true,
        });
        let history = History::new(
            Instant::now(),
            Snapshot {
                active: 1,
                ..Default::default()
            },
        );
        (view, metrics, history)
    }

    #[test]
    fn intervals_weight_measured_time_and_request_reuse_keeps_separate_histories() {
        let (mut view, _, _) = fixture();
        let totals = view.totals(3);
        assert_eq!(
            totals.iter().map(|t| t.gpu_ns).collect::<Vec<_>>(),
            vec![20_000_000, 10_000_000, 1_000_000]
        );
        assert_eq!(
            (totals[1].calls, totals[1].timed, totals[1].lanes),
            (3, 2, 3)
        );
        let requests = view.requests();
        assert_eq!(
            requests,
            vec![
                RequestHandle {
                    slot: 0,
                    generation: 2
                },
                RequestHandle {
                    slot: 0,
                    generation: 1
                }
            ]
        );
        let text = |spans: Vec<Span<'_>>| {
            spans
                .into_iter()
                .map(|s| s.content.into_owned())
                .collect::<String>()
        };
        assert_eq!(
            text(view.activity(requests[0], 10, &catalog())),
            "·········S"
        );
        assert_eq!(
            text(view.activity(requests[1], 10, &catalog())),
            "·······PP·"
        );
        view.scroll(100);
        assert_eq!(view.offset, 1);
        view.scroll(-100);
        assert_eq!(view.offset, 0);
    }

    #[test]
    fn scheduler_pane_renders_at_terminal_sizes_with_legible_units_and_no_nan() {
        let (view, metrics, history) = fixture();
        for (width, height) in [(150, 38), (100, 30), (80, 24), (40, 12), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| {
                    render_pane(frame, "Qwen", &history, &view, &metrics, Pane::Scheduler)
                })
                .unwrap();
            let buffer = terminal.backend().buffer();
            let text: String = buffer.content.iter().map(|c| c.symbol()).collect();
            assert!(!text.contains("NaN"));
            assert!(!text.contains("inf"));
            if width >= 80 {
                for label in [
                    "Overview",
                    "Scheduler",
                    "prefill b1 t1024",
                    "speculative b4 d5",
                    "? (learning)",
                    "0.1",
                    "0.2",
                    "retired",
                    "1 untimed",
                ] {
                    assert!(text.contains(label), "missing {label} at {width}x{height}");
                }
            }
            if width >= 150 {
                assert!(text.contains("64.5%"));
                assert!(text.contains("Bursts"));
            }
        }
    }

    #[test]
    fn idle_time_expires_graph_rows_and_finished_requests() {
        let (mut view, metrics, _) = fixture();
        metrics.scheduler_trace.start();
        view.sample(&metrics.scheduler_trace, 30_000_000_000);
        assert!(view.samples.is_empty());
        assert!(view.totals(3).is_empty());
        assert!(view.requests().is_empty());
    }
}
