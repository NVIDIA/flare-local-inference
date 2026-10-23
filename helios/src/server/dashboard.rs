//! Terminal rendering consumes bounded telemetry independently of execution.

mod scheduler;

use std::{
    collections::VecDeque,
    io::{self, IsTerminal},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Paragraph},
};

use super::metrics::{Metrics, Snapshot};

const REFRESH: Duration = Duration::from_millis(500);
const WINDOW: Duration = Duration::from_secs(2);
const HISTORY: f64 = 120.0;
pub const BACKGROUND: Color = Color::Rgb(13, 18, 28);
pub const MUTED: Color = Color::Rgb(126, 146, 166);
pub const TEXT: Color = Color::Rgb(221, 232, 242);
pub const CYAN: Color = Color::Rgb(84, 214, 223);
pub const GREEN: Color = Color::Rgb(145, 222, 139);
pub const PURPLE: Color = Color::Rgb(191, 157, 245);
pub const GOLD: Color = Color::Rgb(244, 196, 112);
static VISIBLE: AtomicBool = AtomicBool::new(false);

/// Telemetry and rendering pieces of the serving dashboard for dashboards
/// outside the serving process, such as one fed by `GET /metrics?format=json`.
/// The rendering functions take `ratatui` types; `ratatui` and `crossterm` are
/// re-exported so dependents render with the same versions.
pub mod view {
    pub use super::{
        BACKGROUND, CYAN, GOLD, GREEN, History, MUTED, PURPLE, Rates, Series, TEXT, gibibytes,
        kv_cache_lines, plot, rate, weight_bandwidth,
    };
    pub use crate::server::metrics::Snapshot;
    pub use {crossterm, ratatui};
}

pub(super) fn restore_for_exit() {
    if VISIBLE.load(Ordering::Relaxed) {
        let leave = b"\x1b[?1049l";
        // SAFETY: Signal handlers may only use async-signal-safe terminal writes.
        #[cfg(unix)]
        unsafe {
            libc::write(libc::STDERR_FILENO, leave.as_ptr().cast(), leave.len());
        }
        #[cfg(windows)]
        {
            let _ = std::io::Write::write_all(&mut io::stderr(), leave);
        }
    }
}

pub(super) struct Dashboard {
    stop: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Dashboard {
    pub fn start(
        metrics: Arc<Metrics>,
        title: String,
        learned_drafter: bool,
    ) -> io::Result<Option<Self>> {
        if !io::stderr().is_terminal()
            || std::env::var("TERM").is_ok_and(|term| term == "dumb")
            || VISIBLE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
        {
            return Ok(None);
        }
        let (stop, receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("flare-dashboard".into())
            .spawn(move || {
                // Linux priorities are per-thread; rendering yields to serving under CPU pressure.
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, 0, 10);
                }
                let _visible = Visible;
                let result = (|| -> io::Result<()> {
                    let _screen = Screen;
                    let input = InputMode::open()?;
                    execute!(io::stderr(), EnterAlternateScreen, Hide)?;
                    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
                    terminal.clear()?;
                    let start = Instant::now();
                    let mut history = History::new(start, metrics.snapshot(learned_drafter));
                    let mut scheduler = scheduler::SchedulerView::default();
                    let mut pane = Pane::Overview;
                    let _telemetry = Telemetry(&metrics);
                    metrics.scheduler_trace.start();
                    let mut sampled = start - REFRESH;
                    let mut redraw = true;
                    loop {
                        let now = Instant::now();
                        if now.duration_since(sampled) >= REFRESH {
                            history.sample(now, metrics.snapshot(learned_drafter));
                            scheduler.sample(
                                &metrics.scheduler_trace,
                                super::observation::monotonic_ns(),
                            );
                            sampled = now;
                            redraw = true;
                        }
                        if input.is_some() {
                            // Limit input processing so a held key cannot postpone shutdown or samples.
                            for _ in 0..32 {
                                if !event::poll(Duration::ZERO)? {
                                    break;
                                }
                                match event::read()? {
                                    Event::Key(key) => {
                                        redraw |= navigate(&mut pane, &mut scheduler, key)
                                    }
                                    Event::Resize(..) => redraw = true,
                                    _ => {}
                                }
                            }
                        }
                        if redraw {
                            terminal.draw(|frame| {
                                render_pane(frame, &title, &history, &scheduler, &metrics, pane)
                            })?;
                            redraw = false;
                        }
                        match receiver.recv_timeout(Duration::from_millis(50)) {
                            Err(RecvTimeoutError::Timeout) => {}
                            Ok(()) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                        }
                    }
                })();
                if let Err(error) = result {
                    eprintln!("Flare dashboard stopped: {error}");
                }
            });
        match thread {
            Ok(thread) => Ok(Some(Self {
                stop,
                thread: Some(thread),
            })),
            Err(error) => {
                VISIBLE.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for Dashboard {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Visible;
impl Drop for Visible {
    fn drop(&mut self) {
        VISIBLE.store(false, Ordering::Release);
    }
}

struct Screen;
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = execute!(io::stderr(), Show, LeaveAlternateScreen);
    }
}

struct Telemetry<'a>(&'a Metrics);
impl Drop for Telemetry<'_> {
    fn drop(&mut self) {
        self.0.scheduler_trace.stop();
    }
}

#[cfg(unix)]
struct InputMode {
    fd: libc::c_int,
    original: libc::termios,
}

#[cfg(unix)]
impl InputMode {
    fn open() -> io::Result<Option<Self>> {
        io::stdin()
            .is_terminal()
            .then(|| Self::for_fd(libc::STDIN_FILENO))
            .transpose()
    }

    fn for_fd(fd: libc::c_int) -> io::Result<Self> {
        // Keep ISIG enabled so Ctrl-C follows the server's normal shutdown path.
        unsafe {
            let mut original = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut original) != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut mode = original;
            mode.c_lflag &= !(libc::ICANON | libc::ECHO);
            mode.c_cc[libc::VMIN] = 1;
            mode.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(fd, libc::TCSANOW, &mode) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { fd, original })
        }
    }
}

#[cfg(unix)]
impl Drop for InputMode {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pane {
    Overview,
    Scheduler,
}

fn navigate(pane: &mut Pane, scheduler: &mut scheduler::SchedulerView, key: KeyEvent) -> bool {
    if key.kind == KeyEventKind::Release {
        return false;
    }
    match key.code {
        KeyCode::Left | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => {
            *pane = match pane {
                Pane::Overview => Pane::Scheduler,
                Pane::Scheduler => Pane::Overview,
            };
            true
        }
        KeyCode::Up if *pane == Pane::Scheduler => {
            scheduler.scroll(-1);
            true
        }
        KeyCode::Down if *pane == Pane::Scheduler => {
            scheduler.scroll(1);
            true
        }
        _ => false,
    }
}

fn render_pane(
    frame: &mut Frame,
    title: &str,
    history: &History,
    scheduler: &scheduler::SchedulerView,
    metrics: &Metrics,
    pane: Pane,
) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .split(area);
    render_header(frame, title, history, rows[0]);
    let tab = |name, active| {
        Span::styled(
            name,
            if active {
                Style::default()
                    .fg(BACKGROUND)
                    .bg(CYAN)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(MUTED)
            },
        )
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ← → ", Style::default().fg(MUTED)),
            tab(" Overview ", pane == Pane::Overview),
            Span::raw(" "),
            tab(" Scheduler ", pane == Pane::Scheduler),
        ])),
        rows[1],
    );
    match pane {
        Pane::Overview => render_overview(frame, history, rows[2]),
        Pane::Scheduler => {
            scheduler.render(frame, rows[2], history.snapshot, &metrics.scheduler_trace)
        }
    }
}

/// Per-second rates over the most recent sampling window of up to two seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rates {
    /// Prompt tokens prefilled per second.
    pub prefill: f64,
    /// Generated tokens per second.
    pub generated: f64,
    /// Complete speculative generation programs completed per second.
    pub verify: f64,
    /// Accepted drafts per draft-offering request round; `None` without such rounds.
    pub acceptance: Option<f64>,
    /// Allocated draft positions per progressing generation request round.
    pub depth: Option<f64>,
    /// Weight bytes read per second: complete speculative programs per second times the bytes one decode
    /// step reads, as the package declares; zero when it declares none.
    pub weights: f64,
}

/// Rates derived from successive [`Snapshot`]s, with 120 seconds of rate points.
pub struct History {
    start: Instant,
    previous: VecDeque<(Instant, Snapshot)>,
    points: VecDeque<(f64, Rates)>,
    snapshot: Snapshot,
    rates: Rates,
}

impl History {
    /// Starts a history whose point times count seconds from `start`.
    pub fn new(start: Instant, snapshot: Snapshot) -> Self {
        Self {
            start,
            previous: VecDeque::from([(start, snapshot)]),
            points: VecDeque::new(),
            snapshot,
            rates: Rates::default(),
        }
    }

    /// Records `snapshot` taken at `now` and appends one rate point.
    pub fn sample(&mut self, now: Instant, snapshot: Snapshot) {
        while self.previous.len() > 1 && now.duration_since(self.previous[1].0) >= WINDOW {
            self.previous.pop_front();
        }
        let (then, before) = self.previous[0];
        let elapsed = now.duration_since(then).as_secs_f64();
        let per_second = |after: u64, before: u64| {
            if elapsed > 0.0 {
                after.saturating_sub(before) as f64 / elapsed
            } else {
                0.0
            }
        };
        let rounds = snapshot.draft_rounds.saturating_sub(before.draft_rounds);
        let verifications = snapshot.verification_rounds.saturating_sub(before.verification_rounds);
        self.rates = Rates {
            prefill: per_second(snapshot.prefill_tokens, before.prefill_tokens),
            generated: per_second(snapshot.generated_tokens, before.generated_tokens),
            verify: per_second(snapshot.verify_steps, before.verify_steps),
            depth: (verifications > 0).then(|| {
                snapshot
                    .verification_drafts
                    .saturating_sub(before.verification_drafts) as f64
                    / verifications as f64
            }),
            acceptance: (rounds > 0).then(|| {
                snapshot
                    .accepted_drafts
                    .saturating_sub(before.accepted_drafts) as f64
                    / rounds as f64
            }),
            weights: per_second(snapshot.verify_steps, before.verify_steps)
                * snapshot.decode_weight_bytes as f64,
        };
        let time = now.duration_since(self.start).as_secs_f64();
        self.points.push_back((time, self.rates));
        while self
            .points
            .front()
            .is_some_and(|(at, _)| time - at > HISTORY)
        {
            self.points.pop_front();
        }
        self.previous.push_back((now, snapshot));
        self.snapshot = snapshot;
    }

    /// Most recently sampled snapshot.
    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
    }

    /// Rates at the most recent sample.
    pub fn rates(&self) -> Rates {
        self.rates
    }

    /// Rate points from the last 120 seconds, oldest first, as seconds since
    /// `start` and the rates at that time.
    pub fn points(&self) -> &VecDeque<(f64, Rates)> {
        &self.points
    }
}

/// Formats a rate with one decimal, in thousands with a `k` suffix from 10,000.
pub fn rate(value: f64) -> String {
    if value >= 10_000.0 {
        format!("{:.1}k", value / 1000.0)
    } else {
        format!("{value:.1}")
    }
}

/// Formats a byte count in GiB with one decimal.
pub fn gibibytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Weight bandwidth as `N GB/s`, or a dash when the package declares no decode weight bytes.
pub fn weight_bandwidth(s: Snapshot, r: Rates) -> String {
    if s.decode_weight_bytes == 0 {
        "—".into()
    } else {
        format!("{:.0} GB/s", r.weights / 1e9)
    }
}

/// Muted summary line of cached, free, and total KV capacity in tokens.
pub fn kv_cache_lines(s: Snapshot) -> [Line<'static>; 1] {
    [
        // Line::from(format!(
        //     " KV PAGES {} cached / {} free / {} total",
        //     s.cached_pages, s.free_pages, s.total_pages
        // )),
        Line::from(format!(
            " KV TOKENS {} cached / {} free / {} total",
            s.cached_pages.saturating_mul(s.page_tokens),
            s.free_pages.saturating_mul(s.page_tokens),
            s.total_pages.saturating_mul(s.page_tokens)
        )),
    ]
    .map(|line| line.style(Style::default().fg(MUTED)))
}

fn render_header(frame: &mut Frame, title: &str, history: &History, area: Rect) {
    let s = history.snapshot;
    let age = history.points.back().map_or(0, |(t, _)| *t as u64);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " FLARE ",
                Style::default()
                    .fg(BACKGROUND)
                    .bg(CYAN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" / HELIOS  ", Style::default().fg(CYAN)),
            Span::styled(
                if s.active > 0 {
                    "● RUNNING"
                } else {
                    "● READY"
                },
                Style::default().fg(GREEN),
            ),
            Span::styled(
                format!("    {:02}:{:02}:{:02}", age / 3600, age / 60 % 60, age % 60),
                Style::default().fg(MUTED),
            ),
            Span::styled(format!("    {title}"), Style::default().fg(MUTED)),
        ])),
        area,
    );
}

fn render_overview(frame: &mut Frame, history: &History, area: Rect) {
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    let s = history.snapshot;
    let r = history.rates;
    let acceptance = r
        .acceptance
        .map_or_else(|| "—".into(), |v| format!("{v:.2}"));
    let depth = r.depth.map_or_else(|| "—".into(), |v| format!("{v:.2}"));
    let [kv_tokens] = kv_cache_lines(s);
    if area.width < 64 || area.height < 22 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(format!(" Prefill      {} tok/s", rate(r.prefill))),
                Line::from(format!(" Generation   {} tok/s", rate(r.generated))),
                Line::from(format!(" Verify       {} steps/s", rate(r.verify))),
                Line::from(format!(" Acceptance   {acceptance} drafts/round")),
                Line::from(format!(" Verify depth {depth} drafts/step")),
                Line::from(format!(" Weights      {}", weight_bandwidth(s, r))),
                Line::from(format!(" Active {}  Queued {}", s.active, s.pending)),
                kv_tokens,
                Line::from(" Enlarge terminal for history plots.")
                    .style(Style::default().fg(MUTED)),
            ]),
            area,
        );
        return;
    }
    let rows = Layout::vertical([
        Constraint::Min(8),
        Constraint::Min(8),
        Constraint::Length(4),
    ])
    .split(area);
    let top =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[0]);
    let bottom =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[1]);
    plot(
        frame,
        top[0],
        &format!(" PREFILL  {} tok/s ", rate(r.prefill)),
        history,
        &[("", CYAN, |r| r.prefill)],
    );
    plot(
        frame,
        top[1],
        &format!(" GENERATION  {} tok/s ", rate(r.generated)),
        history,
        &[("", GREEN, |r| r.generated)],
    );
    plot(
        frame,
        bottom[0],
        &format!(" VERIFY  {} steps/s ", rate(r.verify)),
        history,
        &[("", PURPLE, |r| r.verify)],
    );
    plot(
        frame,
        bottom[1],
        &format!(" ACCEPTANCE {acceptance} / DEPTH {depth} "),
        history,
        &[
            ("accepted", GOLD, |r| r.acceptance.unwrap_or(0.0)),
            ("depth", CYAN, |r| r.depth.unwrap_or(0.0)),
        ],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                " ACTIVE {}   QUEUED {}   COMPLETE {}   FAILED {}",
                s.active, s.pending, s.completed, s.failed
            )),
            Line::from(format!(
                " PREFILL {}   GENERATED {}   REUSED {}",
                s.prefill_tokens, s.generated_tokens, s.reused_tokens
            ))
            .style(Style::default().fg(MUTED)),
            kv_tokens,
            Line::from(format!(
                " HOST CACHE {} hits   {} reused   {} in memory / {} on disk   {} out / {} in",
                s.host_hits,
                s.host_reused_tokens,
                s.host_memory_records,
                s.host_disk_records,
                gibibytes(s.host_bytes_out),
                gibibytes(s.host_bytes_in),
            ))
            .style(Style::default().fg(MUTED)),
        ]),
        rows[2],
    );
}

/// Plot series: legend name (empty for none), line color, and rate selector.
pub type Series = (&'static str, Color, fn(Rates) -> f64);

/// Draws `series` from the history's rate points as Braille lines over the last
/// 120 seconds in a rounded block titled `title`, with 15% headroom.
/// The upper label shows the observed maximum across the series.
pub fn plot(frame: &mut Frame, area: Rect, title: &str, history: &History, series: &[Series]) {
    let now = history.points.back().map_or(0.0, |(time, _)| *time);
    let points: Vec<Vec<_>> = series
        .iter()
        .map(|(_, _, value)| {
            history
                .points
                .iter()
                .map(|(time, rates)| (time - now, value(*rates)))
                .collect()
        })
        .collect();
    let peak = points
        .iter()
        .flatten()
        .map(|(_, value)| *value)
        .fold(0.0, f64::max);
    // Borders, the x-axis and its labels occupy four rows outside the plot.
    let label_rows = area.height.saturating_sub(4).max(2) as usize;
    let mut labels = vec![Line::default(); label_rows];
    labels[label_rows - 1] = rate(peak).into();
    labels[label_rows - 2] = "max".into();
    let datasets = series
        .iter()
        .zip(&points)
        .map(|((name, color, _), data)| {
            let dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(data);
            if name.is_empty() {
                dataset
            } else {
                dataset.name(*name)
            }
        })
        .collect();
    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(MUTED))
                    .title(title.to_owned())
                    .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            )
            .style(Style::default().bg(BACKGROUND))
            .x_axis(
                Axis::default()
                    .bounds([-HISTORY, 0.0])
                    .labels(["−120s", "now"])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([0.0, peak.max(1.0) * 1.15])
                    .labels(labels)
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    #[test]
    fn panes_share_one_status_header_above_navigation() {
        let start = Instant::now();
        let mut history = History::new(start, Snapshot::default());
        history.sample(start + Duration::from_secs(23), Snapshot::default());
        let metrics = Metrics::default();
        let scheduler = scheduler::SchedulerView::default();
        let title = "deepseek-ai/DeepSeek-V4.1-Flash · 127.0.0.1:8001";
        let mut headers = Vec::new();
        for pane in [Pane::Overview, Pane::Scheduler] {
            let mut terminal = Terminal::new(TestBackend::new(150, 35)).unwrap();
            terminal
                .draw(|frame| render_pane(frame, title, &history, &scheduler, &metrics, pane))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let lines: Vec<String> = buffer
                .content
                .chunks(150)
                .map(|line| line.iter().map(|cell| cell.symbol()).collect())
                .collect();
            assert!(lines[0].contains("FLARE  / HELIOS  ● READY    00:00:23"));
            assert!(lines[0].contains(title));
            assert!(lines[1].contains("Overview") && lines[1].contains("Scheduler"));
            assert_eq!(lines.join("\n").matches(title).count(), 1);
            headers.push(lines[0].clone());
        }
        assert_eq!(headers[0], headers[1]);
    }

    #[test]
    fn arrows_switch_panes_without_treating_key_release_as_another_press() {
        let mut pane = Pane::Overview;
        let mut scheduler = scheduler::SchedulerView::default();
        let mut key = KeyEvent::new(KeyCode::Right, event::KeyModifiers::NONE);
        assert!(navigate(&mut pane, &mut scheduler, key));
        assert_eq!(pane, Pane::Scheduler);
        key.kind = KeyEventKind::Release;
        assert!(!navigate(&mut pane, &mut scheduler, key));
        assert_eq!(pane, Pane::Scheduler);
        key = KeyEvent::new(KeyCode::Left, event::KeyModifiers::NONE);
        assert!(navigate(&mut pane, &mut scheduler, key));
        assert_eq!(pane, Pane::Overview);
    }

    #[test]
    #[cfg(unix)]
    fn terminal_input_retains_signals_and_restores_the_original_mode() {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        unsafe {
            let (mut master, mut slave) = (-1, -1);
            assert_eq!(
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null()
                ),
                0
            );
            let _master = OwnedFd::from_raw_fd(master);
            let slave = OwnedFd::from_raw_fd(slave);
            let mut original: libc::termios = std::mem::zeroed();
            assert_eq!(libc::tcgetattr(slave.as_raw_fd(), &mut original), 0);
            let mode = InputMode::for_fd(slave.as_raw_fd()).unwrap();
            let mut active: libc::termios = std::mem::zeroed();
            assert_eq!(libc::tcgetattr(slave.as_raw_fd(), &mut active), 0);
            assert_eq!(active.c_lflag & (libc::ICANON | libc::ECHO), 0);
            assert_eq!(active.c_lflag & libc::ISIG, original.c_lflag & libc::ISIG);
            drop(mode);
            assert_eq!(libc::tcgetattr(slave.as_raw_fd(), &mut active), 0);
            assert_eq!(active.c_lflag, original.c_lflag);
            assert_eq!(active.c_cc, original.c_cc);
        }
    }

    #[test]
    fn rates_use_elapsed_time_and_weight_acceptance_by_request_rounds() {
        let start = Instant::now();
        let mut history = History::new(start, Snapshot::default());
        let first = Snapshot {
            prefill_tokens: 100,
            generated_tokens: 8,
            draft_steps: 7,
            verify_steps: 1,
            verification_drafts: 7,
            verification_rounds: 3,
            accepted_drafts: 6,
            draft_rounds: 2,
            ..Default::default()
        };
        history.sample(start + Duration::from_secs(1), first);
        let last = Snapshot {
            prefill_tokens: 300,
            generated_tokens: 10,
            draft_steps: 8,
            verify_steps: 2,
            verification_drafts: 10,
            verification_rounds: 5,
            accepted_drafts: 7,
            draft_rounds: 3,
            ..Default::default()
        };
        history.sample(start + Duration::from_secs(2), last);
        assert_eq!(history.rates.prefill, 150.0);
        assert_eq!(history.rates.generated, 5.0);
        assert_eq!(history.rates.depth, Some(2.0));
        assert_eq!(history.rates.verify, 1.0);
        assert_eq!(history.rates.acceptance, Some(7.0 / 3.0));
        for tick in 3..=130 {
            history.sample(start + Duration::from_secs(tick), last);
        }
        assert_eq!(history.rates.generated, 0.0);
        assert_eq!(history.rates.acceptance, None);
        assert_eq!(history.rates.depth, None);
        assert!(history.points.len() <= 121);
        assert!(history.previous.len() <= 3);
        history.sample(
            start + Duration::from_secs(131),
            Snapshot {
                verify_steps: last.verify_steps + 1,
                verification_rounds: last.verification_rounds + 2,
                ..last
            },
        );
        assert_eq!(history.rates.depth, Some(0.0));
    }

    #[test]
    fn terminal_renders_braille_and_handles_resize_without_nonfinite_rates() {
        let start = Instant::now();
        let metrics = Metrics::new(100, 128, 0, 0);
        metrics.scheduler(crate::scheduler::SchedulerStats {
            cached_pages: 3,
            free_pages: 20,
            ..Default::default()
        });
        let cache = metrics.snapshot(false);
        let mut history = History::new(start, cache);
        history.sample(start, cache);
        for tick in 1..=120 {
            history.sample(
                start + Duration::from_secs(tick),
                Snapshot {
                    generated_tokens: tick * tick,
                    prefill_tokens: tick * 100,
                    draft_steps: tick * 12,
                    verify_steps: tick * 3,
                    verification_drafts: tick * 12,
                    verification_rounds: tick * 6,
                    accepted_drafts: tick * 7,
                    draft_rounds: tick * 3,
                    ..cache
                },
            );
        }
        for (width, height) in [(120, 35), (80, 24), (80, 20), (40, 12), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| render_overview(frame, &history, frame.area()))
                .unwrap();
            let content: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(!content.contains("NaN"));
            assert!(!content.contains("inf"));
            assert!(!content.contains("KV PAGES"));
            if width >= 80 {
                assert!(content.contains("KV TOKENS 384 cached / 2560 free / 12800 total"));
            }
            if width >= 80 && height >= 22 {
                assert!(content.contains("PREFILL"));
                assert!(content.contains("ACCEPTANCE"));
                assert!(
                    content
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                );
            }
        }
    }
}

#[cfg(windows)]
struct InputMode {
    handle: windows_sys::Win32::Foundation::HANDLE,
    original: u32,
}
#[cfg(windows)]
impl InputMode {
    fn open() -> io::Result<Option<Self>> {
        use windows_sys::Win32::System::Console::*;
        if !io::stdin().is_terminal() {
            return Ok(None);
        }
        unsafe {
            let handle = GetStdHandle(STD_INPUT_HANDLE);
            let mut original = 0;
            if GetConsoleMode(handle, &mut original) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mode =
                (original | ENABLE_PROCESSED_INPUT) & !(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
            if SetConsoleMode(handle, mode) == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Some(Self { handle, original }))
        }
    }
}
#[cfg(windows)]
impl Drop for InputMode {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::System::Console::SetConsoleMode(self.handle, self.original);
        }
    }
}
