//! Animated startup banner and loading rows with measured progress and elapsed time.

mod banner;

use std::{
    io::{IsTerminal, Write},
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Displays one loading stage at a time on stderr; redirected output is plain text.
pub struct StartupProgress {
    enabled: bool,
    terminal: bool,
    output: Arc<Mutex<Output>>,
    animation: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

impl StartupProgress {
    pub fn stderr() -> Self {
        let mut progress = Self::stderr_rows();
        progress.animate_banner();
        progress
    }

    /// Displays progress rows without the Flare startup banner.
    pub fn stderr_rows() -> Self {
        Self {
            enabled: true,
            terminal: std::io::stderr().is_terminal(),
            output: Arc::new(Mutex::new(Output::default())),
            animation: None,
        }
    }

    pub fn silent() -> Self {
        Self {
            enabled: false,
            terminal: false,
            output: Arc::new(Mutex::new(Output::default())),
            animation: None,
        }
    }

    fn animate_banner(&mut self) {
        if !self.terminal {
            return;
        }
        let (width, height) = terminal_size();
        if cfg!(windows) || banner::frame(0.0, width).lines().count() + 2 >= height {
            let mut out = std::io::stderr().lock();
            let _ = out.write_all(banner::frame(1.0, width).as_bytes());
            let _ = out.flush();
            return;
        }
        let started = Instant::now();
        let mut output = self.output.lock().unwrap();
        output.started = Some(started);
        output.redraw(0.0);
        drop(output);
        let output = self.output.clone();
        let (stop, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            for tick in 1.. {
                let stopped = !matches!(
                    receiver.recv_timeout(
                        (started + (Duration::from_micros(62_500) * tick).min(banner::DURATION))
                            .saturating_duration_since(Instant::now()),
                    ),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                let mut output = output.lock().unwrap();
                let progress = output.progress();
                if stopped || progress >= 1.0 {
                    output.redraw(1.0);
                    output.started = None;
                    output.completed.clear();
                    output.active = None;
                    output.drawn_lines = 0;
                    break;
                }
                output.redraw(progress);
            }
        });
        self.animation = Some((stop, worker));
    }

    /// Settles the header before another terminal UI takes ownership of stderr.
    pub fn finish_animation(&mut self) {
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(());
            let _ = worker.join();
        }
    }

    pub fn phase(&self, label: &str, total: Option<u64>) -> Phase {
        let state = Arc::new(Mutex::new(State {
            label: label.to_owned(),
            detail: String::new(),
            started: Instant::now(),
            elapsed: None,
            completed: 0,
            total,
        }));
        let mut phase = Phase {
            state: state.clone(),
            enabled: self.enabled,
            terminal: self.terminal,
            output: self.output.clone(),
            finished: false,
            animation: None,
        };
        if self.enabled {
            self.output.lock().unwrap().write(
                &state.lock().unwrap(),
                self.terminal,
                Status::Running,
            );
        }
        if self.terminal {
            let (stop, receiver) = mpsc::channel();
            let output = self.output.clone();
            let worker = thread::spawn(move || {
                while matches!(
                    receiver.recv_timeout(Duration::from_millis(125)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    output
                        .lock()
                        .unwrap()
                        .write(&state.lock().unwrap(), true, Status::Running);
                }
            });
            phase.animation = Some((stop, worker));
        }
        phase
    }
}

impl Drop for StartupProgress {
    fn drop(&mut self) {
        self.finish_animation();
    }
}

/// A stage prints FAILED if dropped without an explicit successful finish.
pub struct Phase {
    state: Arc<Mutex<State>>,
    enabled: bool,
    terminal: bool,
    output: Arc<Mutex<Output>>,
    finished: bool,
    animation: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

impl Phase {
    pub fn advance(&self, count: u64) {
        let mut state = self.state.lock().unwrap();
        state.completed = state.completed.saturating_add(count);
    }

    pub fn advance_bytes(&self, count: u64) {
        let mut state = self.state.lock().unwrap();
        state.completed = state.completed.saturating_add(count);
        state.detail = format!(
            "{:.1} / {:.1} GiB",
            state.completed as f64 / (1u64 << 30) as f64,
            state.total.unwrap_or(0) as f64 / (1u64 << 30) as f64,
        );
    }

    pub fn detail(&self, detail: impl Into<String>) {
        self.state.lock().unwrap().detail = detail.into();
    }

    pub fn finish(mut self, detail: impl Into<String>) {
        self.detail(detail);
        self.finished = true;
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(());
            let _ = worker.join();
        }
        if self.enabled {
            self.output.lock().unwrap().write(
                &self.state.lock().unwrap(),
                self.terminal,
                if self.finished {
                    Status::Ok
                } else {
                    Status::Failed
                },
            );
        }
    }
}

#[derive(Clone)]
struct State {
    label: String,
    detail: String,
    started: Instant,
    elapsed: Option<Duration>,
    completed: u64,
    total: Option<u64>,
}

#[derive(Clone, Copy)]
enum Status {
    Running,
    Ok,
    Failed,
}

impl State {
    fn bar(&self, elapsed: f64) -> String {
        use std::fmt::Write;

        let mut bar = String::new();
        if let Some(total) = self.total {
            let halves = if total == 0 {
                20
            } else {
                (20.0 * self.completed.min(total) as f64 / total as f64) as usize
            };
            let full = halves / 2;
            let half = halves % 2;
            let color = if self.completed >= total { 32 } else { 36 };
            let _ = write!(bar, "\x1b[{color}m{}", "━".repeat(full));
            if half != 0 {
                bar.push('╸');
            }
            let mut remaining = 10 - full - half;
            if remaining != 0 {
                bar.push_str("\x1b[38;2;58;58;58m");
                if half == 0 && full != 0 {
                    bar.push('╺');
                    remaining -= 1;
                }
                bar.push_str(&"━".repeat(remaining));
            }
        } else {
            let offset = (20 - (elapsed * 15.0) as usize % 20) % 20;
            for index in offset..offset + 10 {
                let fade = 0.5 + (index as f64 / 20.0 * std::f64::consts::TAU).cos() / 2.0;
                let red = (58.0 * fade) as u8;
                let cyan = (128.0 * (1.0 - fade) + 58.0 * fade) as u8;
                let _ = write!(bar, "\x1b[38;2;{red};{cyan};{cyan}m━");
            }
        }
        bar
    }

    fn render(&self, terminal: bool, status: Status) -> String {
        use std::fmt::Write;
        let elapsed = self
            .elapsed
            .unwrap_or_else(|| self.started.elapsed())
            .as_secs_f64();
        let status_text = match status {
            Status::Ok => "    OK    ".to_owned(),
            Status::Failed => "  FAILED  ".to_owned(),
            Status::Running if terminal => self.bar(elapsed),
            Status::Running => " PENDING  ".to_owned(),
        };
        let label = format!("] {}.", self.label);
        let detail = if self.detail.is_empty() {
            String::new()
        } else {
            format!("  {}", self.detail)
        };
        let timing = format!("  {elapsed:.1}s");
        let width = if terminal {
            terminal_width()
        } else {
            usize::MAX
        };
        let available = width.saturating_sub(11 + timing.chars().count());
        let body = truncate(&(label.clone() + &detail), available);
        let label_end = body
            .char_indices()
            .nth(label.chars().count())
            .map_or(body.len(), |(i, _)| i);
        let (label, detail) = body.split_at(label_end);
        let padding = if terminal {
            available.saturating_sub(body.chars().count())
        } else {
            0
        };
        let mut out = String::new();
        if terminal {
            let color = match status {
                Status::Running => "36",
                Status::Ok => "1;32",
                Status::Failed => "1;31",
            };
            let _ = write!(
                out,
                "\r\x1b[2K[\x1b[{color}m{status_text}\x1b[0m{label}\x1b[2m{detail}{:padding$}{timing}\x1b[0m",
                ""
            );
        } else {
            let _ = write!(out, "[{status_text}{label}{detail}{timing}");
        }
        out
    }
}

#[derive(Default)]
struct Output {
    started: Option<Instant>,
    completed: Vec<(State, Status)>,
    active: Option<State>,
    drawn_lines: usize,
}

impl Output {
    fn progress(&self) -> f64 {
        self.started.map_or(1.0, |started| {
            (started.elapsed().as_secs_f64() / banner::DURATION.as_secs_f64()).min(1.0)
        })
    }

    fn write(&mut self, state: &State, terminal: bool, status: Status) {
        if self.started.is_some() {
            if matches!(status, Status::Running) {
                self.active = Some(state.clone());
            } else {
                let mut state = state.clone();
                state.elapsed = Some(state.started.elapsed());
                self.completed.push((state, status));
                self.active = None;
            }
            self.redraw(self.progress());
        } else {
            let mut out = std::io::stderr().lock();
            let _ = out.write_all(state.render(terminal, status).as_bytes());
            if !terminal || !matches!(status, Status::Running) {
                let _ = writeln!(out);
            }
            let _ = out.flush();
        }
    }

    fn redraw(&mut self, progress: f64) {
        if self.started.is_none() {
            return;
        }
        let (width, height) = terminal_size();
        let mut content = banner::frame(progress, width);
        for (state, status) in &self.completed {
            content.push_str(&state.render(true, *status));
            content.push('\n');
        }
        if let Some(state) = &self.active {
            content.push_str(&state.render(true, Status::Running));
        }
        let lines: Vec<_> = content.split('\n').collect();
        let visible = &lines[lines.len().saturating_sub(height.saturating_sub(1).max(1))..];
        let mut out = std::io::stderr().lock();
        if self.drawn_lines > 1 {
            let _ = write!(
                out,
                "\x1b[{}A",
                (self.drawn_lines - 1).min(height.saturating_sub(1))
            );
        }
        let _ = write!(out, "\r\x1b[J{}", visible.join("\n"));
        let _ = out.flush();
        self.drawn_lines = visible.len();
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    text.chars().take(width - 1).collect::<String>() + "…"
}

fn terminal_width() -> usize {
    terminal_size().0
}

fn terminal_size() -> (usize, usize) {
    crate::os::terminal_size()
}
