//! Signed model metadata with a compact summary and selectable detail sections.

use std::{fmt::Write as _, io::IsTerminal};

use flare_helios::runtime_package::{
    InspectionOptions, PackageClaim, VerifiedPackage, unverified_metadata,
};
use serde_json::Value;

use crate::cli::Inspect;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn run(args: Inspect) -> Result<()> {
    let options = InspectionOptions {
        memory_map: args.memory_map,
        graphs: args.graphs,
        frontend: args.frontend,
        all: args.all,
    };
    let mut value = if args.unverified {
        unverified_metadata(&args.package, options, args.rank)?
    } else {
        verified(&args, options)?
    };
    value["package"]["path"] = Value::String(args.package.display().to_string());
    if args.json {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    let stdout = std::io::stdout();
    let color =
        !args.plain && stdout.is_terminal() && !std::env::var("TERM").is_ok_and(|t| t == "dumb");
    let width = if stdout.is_terminal() {
        terminal_width().clamp(MIN_WIDTH, MAX_WIDTH)
    } else {
        DEFAULT_WIDTH
    };
    print!("{}", render(&value, width, color));
    Ok(())
}

fn verified(args: &Inspect, options: InspectionOptions) -> Result<Value> {
    let root = args
        .package
        .canonicalize()
        .map_err(|e| format!("cannot access package {}: {e}", args.package.display()))?;
    let claim = PackageClaim::read(&root)?;
    let mut profiles = crate::trust::publishers(
        &args.publisher_keys,
        claim.development,
        crate::environment::home_directory,
    )?;
    if !profiles.iter().any(|p| p.fingerprint() == claim.signer) {
        if let Some(profile) =
            crate::trust::remembered_profile(&claim, args.trust_store.as_deref())?
        {
            profiles.push(profile);
        }
    }
    let package = VerifiedPackage::open(&root, &profiles, args.rank).map_err(
        |error| -> Box<dyn std::error::Error> {
            if error.to_string().contains("unknown package publisher") {
                return format!(
                    "cannot verify signing key {}: no embedded, remembered, or --publisher-keys \
                     profile names it (publisher {} at {})",
                    claim.signer, claim.namespace, claim.official_source
                )
                .into();
            }
            error.into()
        },
    )?;
    Ok(package.describe(options)?)
}

const MIN_WIDTH: usize = 64;
const MAX_WIDTH: usize = 120;
const DEFAULT_WIDTH: usize = 80;
const KEY_WIDTH: usize = 24;

fn terminal_width() -> usize {
    #[cfg(unix)]
    {
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: TIOCGWINSZ fills a winsize for a terminal descriptor; failure leaves zeros.
        if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
            && size.ws_col > 0
        {
            return size.ws_col as usize;
        }
    }
    DEFAULT_WIDTH
}

fn visible(text: &str) -> usize {
    let mut count = 0;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            count += 1;
        }
    }
    count
}

fn model_letter(letter: char) -> Option<[&'static str; 5]> {
    Some(match letter.to_ascii_uppercase() {
        'A' => ["▄██▄ ", "█▀▀█╗", "█  █║", "█  █║", "╚  ╚╝"],
        'B' => ["███▄ ", "█▄▄█╗", "█▀▀█╗", "█▄▄█║", "╚═══╝"],
        'C' => ["▄███╗", "█╔══╝", "█║   ", "▀███╗", " ╚══╝"],
        'D' => ["███▄ ", "█  █╗", "█  █║", "█▄▄█║", "╚═══╝"],
        'E' => ["███╗", "█▄▄╝", "█▀▀ ", "███╗", "╚══╝"],
        'F' => ["███╗", "█▄▄╝", "█▀▀ ", "█║  ", "╚╝  "],
        'G' => ["▄███╗", "█╔══╝", "█ ▀█╗", "▀███║", " ╚══╝"],
        'H' => ["█  █╗", "█▄▄█║", "█▀▀█║", "█  █║", "╚  ╚╝"],
        'I' => ["████╗", " █╔═╝", " █║  ", "████╗", "╚═══╝"],
        'J' => ["   █╗", "   █║", "█  █║", "▀██▀╝", " ╚═╝ "],
        'K' => ["█  █╗", "█▄█▀╝", "█▀█▄ ", "█  █╗", "╚  ╚╝"],
        'L' => ["█╗  ", "█║  ", "█║  ", "███╗", "╚══╝"],
        'M' => ["█▄▄█╗", "████║", "█▀▀█║", "█  █║", "╚  ╚╝"],
        'N' => ["█▄ █╗", "██▄█║", "█▀██║", "█ ▀█║", "╚  ╚╝"],
        'O' => ["▄██▄ ", "█  █╗", "█  █║", "▀██▀╝", " ╚═╝ "],
        'P' => ["███▄ ", "█▄▄█╗", "█▀▀▀╝", "█║   ", "╚╝   "],
        'Q' => ["▄██▄ ", "█  █╗", "█ ▄█║", "▀███╗", " ╚══╝"],
        'R' => ["███▄ ", "█▄▄█╗", "█▀█▀╝", "█  █╗", "╚  ╚╝"],
        'S' => ["▄███╗", "█▄▄▄╝", "  ▀█╗", "███▀╝", "╚══╝ "],
        'T' => ["████╗", " █╔═╝", " █║  ", " █║  ", " ╚╝  "],
        'U' => ["█  █╗", "█  █║", "█  █║", "▀██▀╝", " ╚═╝ "],
        'V' => ["█  █╗", "█  █║", "▀▄▄▀╝", " ██╗ ", " ╚═╝ "],
        'W' => ["█  █╗", "█  █║", "████║", "▀██▀╝", " ╚═╝ "],
        'X' => ["█  █╗", "▀██▀╝", "▄██▄ ", "█  █╗", "╚  ╚╝"],
        'Y' => ["█  █╗", "▀██▀╝", " ██╗ ", " ██║ ", " ╚═╝ "],
        'Z' => ["████╗", " ▄█▀╝", "▄█▀  ", "████╗", "╚═══╝"],
        '0' => ["▄██▄ ", "█  █╗", "█  █║", "▀██▀╝", " ╚═╝ "],
        '1' => ["▄██╗", " ██║", " ██║", " ██║", " ╚═╝"],
        '2' => ["███▄ ", " ▄▄█╗", "▄█▀▀╝", "████╗", "╚═══╝"],
        '3' => ["███▄ ", " ▄▄█╗", " ▀▀█╗", "███▀╝", "╚══╝ "],
        '4' => ["█  █╗", "█▄▄█║", "▀▀▀█║", "   █║", "   ╚╝"],
        '5' => ["████╗", "█▄▄▄╝", "  ▀█╗", "███▀╝", "╚══╝ "],
        '6' => ["▄███╗", "█▄▄▄╝", "█  █╗", "▀██▀╝", " ╚═╝ "],
        '7' => ["████╗", " ▄█▀╝", " ██║ ", " ██║ ", " ╚═╝ "],
        '8' => ["▄██▄ ", "█▄▄█╗", "█  █║", "▀██▀╝", " ╚═╝ "],
        '9' => ["▄██▄ ", "█▄▄█╗", " ▀▀█║", "███▀╝", "╚══╝ "],
        '-' | ' ' => [" ", " ", " ", " ", " "],
        '_' => ["   ", "   ", "   ", "██╗", "╚═╝"],
        '.' => [" ", " ", " ", "▄", "▀"],
        '/' => [" ▄▀", " █ ", "▄▀ ", "█  ", "▀  "],
        _ => return None,
    })
}

fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

fn bytes(value: u64) -> String {
    const KIB: f64 = 1024.0;
    let v = value as f64;
    if v >= KIB * KIB * KIB {
        format!("{:.2} GiB", v / (KIB * KIB * KIB))
    } else if v >= KIB * KIB {
        format!("{:.1} MiB", v / (KIB * KIB))
    } else if v >= KIB {
        format!("{:.0} KiB", v / KIB)
    } else {
        format!("{value} B")
    }
}

fn duration(nanoseconds: u64) -> String {
    if nanoseconds == 0 {
        "–".into()
    } else if nanoseconds >= 1_000_000 {
        format!("{:.2} ms", nanoseconds as f64 / 1e6)
    } else {
        format!("{:.0} µs", nanoseconds as f64 / 1e3)
    }
}

fn decimal(value: f64) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() {
        "0".into()
    } else {
        text.into()
    }
}

/// A value as text, with control characters shown as escapes so a token such as a marker
/// ending in a newline stays on its row.
fn text(value: &Value) -> String {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let mut output = String::new();
    for c in raw.chars() {
        match c {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(output, "\\u{{{:x}}}", c as u32);
            }
            c => output.push(c),
        }
    }
    output
}

fn u64_of(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}

fn f64_of(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn list(value: &Value) -> String {
    value
        .as_array()
        .map(|items| items.iter().map(text).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

struct Canvas {
    out: String,
    width: usize,
    color: bool,
    panel: Option<(String, usize)>,
}

impl Canvas {
    fn styled(&mut self, style: &str, value: &str) {
        if self.color {
            write!(self.out, "\x1b[{style}m{value}\x1b[0m").unwrap();
        } else {
            self.out.push_str(value);
        }
    }

    fn model_title(&mut self, name: &str) {
        self.out.push('\n');
        let letters = name
            .chars()
            .map(|c| model_letter(c).map(|letter| (c, letter)))
            .collect::<Option<Vec<_>>>();
        if let Some(letters) = letters.filter(|letters| !letters.is_empty()) {
            let colors = [
                "1;38;2;255;232;176",
                "1;38;2;255;210;123",
                "1;38;2;255;174;85",
                "1;38;2;255;137;59",
                "1;38;2;230;90;48",
            ];
            let mut start = 0;
            while start < letters.len() {
                let mut end = start;
                let mut used = 0;
                let mut boundary = None;
                while end < letters.len() {
                    let (c, letter) = &letters[end];
                    let width = letter.iter().map(|row| visible(row)).max().unwrap()
                        + if end == start { 0 } else { 2 };
                    if used + width > self.width - 2 {
                        break;
                    }
                    used += width;
                    end += 1;
                    if matches!(c, '-' | '_' | '/' | ' ') {
                        boundary = Some(end);
                    }
                }
                if end < letters.len() {
                    end = boundary.unwrap_or(end);
                }
                if start > 0 {
                    self.out.push('\n');
                }
                for (row, color) in colors.iter().enumerate() {
                    let line = letters[start..end]
                        .iter()
                        .map(|(_, letter)| letter[row])
                        .collect::<Vec<_>>()
                        .join("  ");
                    self.out.push_str("  ");
                    self.styled(color, line.trim_end());
                    self.out.push('\n');
                }
                start = end;
            }
        }
        self.out.push('\n');
    }

    fn content_width(&self) -> usize {
        self.width - if self.panel.is_some() { 4 } else { 0 }
    }

    fn open(&mut self, title: &str) {
        self.close();
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
        self.panel = Some((title.to_owned(), self.out.len()));
    }

    fn close(&mut self) {
        let Some((title, start)) = self.panel.take() else {
            return;
        };
        let content = self.out.split_off(start);
        self.styled("2;36", "╭─ ");
        self.styled("1;36", &title);
        self.styled(
            "2;36",
            &format!(
                " {}╮",
                "─".repeat(self.width.saturating_sub(visible(&title) + 5))
            ),
        );
        self.out.push('\n');
        for line in content.lines() {
            self.styled("2;36", "│ ");
            self.out.push_str(line);
            self.out
                .push_str(&" ".repeat(self.width.saturating_sub(visible(line) + 4)));
            self.styled("2;36", " │");
            self.out.push('\n');
        }
        self.styled("2;36", &format!("╰{}╯", "─".repeat(self.width - 2)));
        self.out.push_str("\n\n");
    }

    /// Lines of at most `width` characters, broken at spaces where there are any.
    fn wrap(value: &str, width: usize) -> Vec<String> {
        let width = width.max(1);
        let mut lines = Vec::new();
        let mut current = String::new();
        for word in value.split(' ') {
            let mut word = word.to_string();
            while visible(&word) > width {
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                }
                let head: String = word.chars().take(width).collect();
                word = word.chars().skip(width).collect();
                lines.push(head);
            }
            let joined = if current.is_empty() {
                word.clone()
            } else {
                format!("{current} {word}")
            };
            if visible(&joined) > width {
                lines.push(std::mem::take(&mut current));
                current = word;
            } else {
                current = joined;
            }
        }
        if !current.is_empty() || lines.is_empty() {
            lines.push(current);
        }
        lines
    }

    fn row(&mut self, key: &str, value: &str) {
        let long_key = visible(key) > KEY_WIDTH;
        if long_key {
            for line in Self::wrap(key, self.content_width() - 2) {
                self.out.push_str("  ");
                self.styled("2", &line);
                self.out.push('\n');
            }
        }
        let room = self.content_width().saturating_sub(KEY_WIDTH + 4);
        for (i, line) in Self::wrap(value, room).iter().enumerate() {
            self.out.push_str("  ");
            if i == 0 && !long_key {
                self.styled("2", &format!("{key:<KEY_WIDTH$}"));
            } else {
                self.out.push_str(&" ".repeat(KEY_WIDTH));
            }
            writeln!(self.out, "  {line}").unwrap();
        }
    }

    fn number(&mut self, key: &str, value: &str) {
        self.row(key, value);
    }

    fn note(&mut self, value: &str) {
        for line in Self::wrap(value, self.content_width() - 2) {
            self.out.push_str("  ");
            self.styled("2", &line);
            self.out.push('\n');
        }
    }

    fn table(&mut self, head: &[&str], rows: &[Vec<String>]) {
        if rows.is_empty() {
            return;
        }
        let columns = head.len();
        let mut widths = head.iter().map(|h| visible(h)).collect::<Vec<_>>();
        for row in rows {
            for (i, cell) in row.iter().enumerate().take(columns) {
                widths[i] = widths[i].max(visible(cell));
            }
        }
        let room = self.content_width() - 2;
        let total = |widths: &[usize]| widths.iter().sum::<usize>() + 2 * (columns - 1);
        while total(&widths) > room {
            let Some((i, _)) = widths
                .iter()
                .enumerate()
                .filter(|(_, w)| **w > 4)
                .max_by_key(|(i, w)| (**w, usize::MAX - i))
            else {
                break;
            };
            widths[i] -= 1;
        }
        let headings = head.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for (row_index, row) in std::iter::once(&headings).chain(rows).enumerate() {
            let cells = row
                .iter()
                .zip(&widths)
                .map(|(cell, width)| Self::wrap(cell, *width))
                .collect::<Vec<_>>();
            let lines = cells.iter().map(Vec::len).max().unwrap_or(0);
            for line in 0..lines {
                let mut rendered = String::from("  ");
                for (i, (cell, width)) in cells.iter().zip(&widths).enumerate() {
                    let value = cell.get(line).map_or("", String::as_str);
                    rendered.push_str(value);
                    if i + 1 < columns {
                        rendered.push_str(&" ".repeat(width.saturating_sub(visible(value)) + 2));
                    }
                }
                if row_index == 0 {
                    self.styled("2", rendered.trim_end());
                    self.out.push('\n');
                } else {
                    writeln!(self.out, "{}", rendered.trim_end()).unwrap();
                }
            }
        }
    }
}

/// Render exactly the sections selected in the inspection document.
pub fn render(value: &Value, width: usize, color: bool) -> String {
    let mut canvas = Canvas {
        out: String::new(),
        width: width.max(MIN_WIDTH),
        color,
        panel: None,
    };
    let package = &value["package"];
    let model = &value["model"];
    let cache = &value["cache"];
    let speculation = &value["speculation"];
    let frontend = &value["frontend"];
    let vision = &value["vision"];
    let engram = &value["engram"];
    let ranks = value["ranks"].as_array().cloned().unwrap_or_default();
    let publisher = &package["publisher"];
    let key_kind = if publisher["development"] == true {
        "development key"
    } else {
        "published key"
    };
    let name = model["served_names"]
        .as_array()
        .and_then(|names| {
            names
                .iter()
                .find(|name| name.as_str().is_some_and(|name| !name.is_empty()))
        })
        .unwrap_or(&model["model_id"]);
    canvas.model_title(&text(name));
    canvas.open("Model");
    canvas.row("name", &text(name));
    canvas.row(
        "publisher",
        &format!(
            "{} ({})",
            text(&publisher["namespace"]),
            if package["signatures_verified"] == true {
                "signatures verified"
            } else {
                "unverified"
            }
        ),
    );
    if let Some(license) = package["license"].as_object() {
        canvas.row("license", &text(&license["name"]));
    }
    let arch = ranks.first().map(|r| &r["architecture"]);
    if let Some(arch) = arch.and_then(Value::as_array).filter(|a| a.len() == 2) {
        canvas.row("architecture", &format!("sm_{}{}", arch[0], arch[1]));
    }
    if model["served_names"]
        .as_array()
        .is_some_and(|names| names.len() > 1)
    {
        canvas.row("served names", &list(&model["served_names"]));
    }
    canvas.row(
        "context",
        &format!(
            "{} tokens",
            thousands(u64_of(&model["max_sequence_tokens"]))
        ),
    );
    canvas.row(
        "prefill chunk",
        &format!("{} tokens", thousands(u64_of(&model["prefill_chunk_size"]))),
    );
    canvas.row(
        "concurrency",
        &format!(
            "{} requests · {} ranks",
            u64_of(&model["max_active"]),
            package["cluster"]["world"]
                .as_u64()
                .unwrap_or(ranks.len() as u64)
        ),
    );
    canvas.row(
        "default output limit",
        &model["max_new_tokens"].as_u64().map_or_else(
            || "remaining context".into(),
            |n| format!("{} tokens", thousands(n)),
        ),
    );
    canvas.row("vocabulary", &thousands(u64_of(&model["vocabulary"])));
    canvas.row(
        "cache histories",
        &list(&cache["history_kinds"])
            .to_lowercase()
            .replace('_', " "),
    );
    canvas.row(
        "speculation",
        &if list(&speculation["widths"]).is_empty() {
            "disabled".into()
        } else {
            format!(
                "{} · widths {}",
                text(&speculation["proposal_source"]),
                list(&speculation["widths"])
            )
        },
    );
    sampling(&mut canvas, &value["sampling_defaults"]);
    canvas.open("Rank resources");
    canvas.table(
        &["Rank", "Device", "Mapped host", "Peer views", "Graphs"],
        &ranks
            .iter()
            .map(|r| {
                vec![
                    text(&r["rank"]),
                    bytes(u64_of(&r["device_bytes"])),
                    bytes(u64_of(&r["mapped_host_bytes"])),
                    bytes(u64_of(&r["peer_bytes"])),
                    text(&r["graphs"]),
                ]
            })
            .collect::<Vec<_>>(),
    );
    canvas.note("Fixed allocations; serving chooses cache backing at load time.");
    if value.get("contract").is_some() {
        canvas.open("PACKAGE");
        canvas.row("file", &text(&package["path"]));
        let companions = package["companions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let companion_bytes: u64 = companions.iter().map(|c| u64_of(&c["bytes"])).sum();
        canvas.number(
            "size",
            &format!(
                "{} primary{}",
                bytes(u64_of(&package["bytes"])),
                if companions.is_empty() {
                    String::new()
                } else {
                    format!(
                        " + {} in {} companion file{}",
                        bytes(companion_bytes),
                        companions.len(),
                        if companions.len() == 1 { "" } else { "s" }
                    )
                }
            ),
        );
        canvas.number(
            "format version",
            &u64_of(&package["format_version"]).to_string(),
        );
        match package["cluster"].as_object() {
            Some(cluster) => {
                canvas.row(
                    "ranks",
                    &format!(
                        "{} on separate hosts, {} verified in this copy",
                        u64_of(&cluster["world"]),
                        ranks.len()
                    ),
                );
                canvas.row("export", &text(&cluster["export_id"]));
            }
            None => canvas.row(
                "ranks",
                &format!("{} on one host (tensor parallel)", ranks.len()),
            ),
        }
        canvas.row(
            "publisher",
            &format!(
                "{} ({})",
                text(&publisher["name"]),
                text(&publisher["namespace"])
            ),
        );
        canvas.row(
            "signing key",
            &format!("{} · {key_kind}", text(&publisher["fingerprint"])),
        );
        canvas.row("source", &text(&package["official_source"]));
        match package["license"].as_object() {
            Some(license) if !text(&license["link"]).is_empty() => canvas.row(
                "license",
                &format!("{} · {}", text(&license["name"]), text(&license["link"])),
            ),
            Some(license) => canvas.row("license", &text(&license["name"])),
            None => canvas.row("license", "not recorded"),
        }
        canvas.close();

        canvas.open("MODEL");
        canvas.row("model id", &text(&model["model_id"]));
        canvas.row("identity", &text(&model["identity"]));
        let served = list(&model["served_names"]);
        canvas.row(
            "served names",
            if served.is_empty() {
                "model id only"
            } else {
                &served
            },
        );
        let vocabulary = u64_of(&model["vocabulary"]);
        let input_vocabulary = u64_of(&model["input_vocabulary"]);
        canvas.number(
            "vocabulary",
            &if input_vocabulary != 0 && input_vocabulary != vocabulary {
                format!(
                    "{} output · {} input",
                    thousands(vocabulary),
                    thousands(input_vocabulary)
                )
            } else {
                thousands(vocabulary)
            },
        );
        if let Some(first) = ranks.first() {
            let arch = first["architecture"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if arch.len() == 2 {
                canvas.row(
                    "gpu architecture",
                    &format!(
                        "sm_{}{} (compute capability {}.{})",
                        u64_of(&arch[0]),
                        u64_of(&arch[1]),
                        u64_of(&arch[0]),
                        u64_of(&arch[1])
                    ),
                );
            }
        }
        let decode_weights = u64_of(&model["decode_weight_bytes"]);
        canvas.number(
            "weights read per step",
            &if decode_weights == 0 {
                "not declared".into()
            } else {
                format!("{} per decode step", bytes(decode_weights))
            },
        );
        canvas.close();

        canvas.open("LIMITS");
        let context = u64_of(&model["max_sequence_tokens"]);
        let max_active = u64_of(&model["max_active"]);
        canvas.number("context", &format!("{} tokens", thousands(context)));
        canvas.number(
            "prefill chunk",
            &format!("{} tokens", thousands(u64_of(&model["prefill_chunk_size"]))),
        );
        canvas.number(
            "concurrency",
            &format!("{} active requests", thousands(max_active)),
        );
        canvas.number(
            "default max tokens",
            &match model["max_new_tokens"].as_u64() {
                Some(n) => format!("{} output tokens", thousands(n)),
                None => "the remaining context".into(),
            },
        );
        if let Some(n) = model["max_body_bytes"].as_u64() {
            canvas.number("request body limit", &bytes(n));
        }
        if let Some(n) = model["queue_capacity"].as_u64() {
            canvas.number("admission queue", &format!("{n} requests"));
        }
        if let Some(n) = model["frontend_workers"].as_u64() {
            canvas.number("frontend workers", &n.to_string());
        }
        if let Some(n) = model["reserve_bytes"].as_u64() {
            canvas.number("gpu reserve", &format!("{} beyond allocations", bytes(n)));
        }
        for (key, name) in [
            ("context_hint", "context hint"),
            ("capacity_hint", "capacity hint"),
        ] {
            let hint = text(&model[key]);
            if !hint.is_empty() {
                canvas.row(name, &hint);
            }
        }
        canvas.close();

        canvas.open("CACHE");
        let page_tokens = u64_of(&cache["page_tokens"]);
        let exported_pages = u64_of(&cache["pages"]);
        let kv_page_bytes = u64_of(&cache["kv_page_bytes"]);
        let state_row_bytes = u64_of(&cache["state_row_bytes"]);
        let max_pages = cache["max_pages"].as_u64();
        let pages = max_pages.unwrap_or(exported_pages);
        let extent = pages.saturating_mul(page_tokens);
        canvas.number("page", &format!("{} tokens", thousands(page_tokens)));
        canvas.number(
            "page extent",
            &format!(
                "{} at most · {} backed at export · serving sizes pages from free memory",
                thousands(pages),
                thousands(exported_pages)
            ),
        );
        canvas.number(
            "kv extent",
            &format!(
                "{} tokens{}",
                thousands(extent),
                if context > 0 {
                    format!(" · {:.2} contexts", extent as f64 / context as f64)
                } else {
                    String::new()
                }
            ),
        );
        if kv_page_bytes > 0 {
            canvas.number(
                "kv bytes",
                &format!(
                    "{} per page · {} per token · selected ranks",
                    bytes(kv_page_bytes),
                    bytes(kv_page_bytes / page_tokens.max(1))
                ),
            );
        }
        canvas.number("state rows", &thousands(u64_of(&cache["state_rows"])));
        if state_row_bytes > 0 {
            canvas.number(
                "state row bytes",
                &format!("{} · selected ranks", bytes(state_row_bytes)),
            );
        }
        canvas.number("snapshots", &thousands(u64_of(&cache["snapshots"])));
        // Identical regions, one per layer, are counted rather than listed.
        let mut regions: Vec<(Vec<String>, usize)> = Vec::new();
        for r in cache["regions"].as_array().into_iter().flatten() {
            let row = vec![
                text(&r["kind"]).to_lowercase().replace('_', " "),
                thousands(u64_of(&r["tokens_per_record"])),
                match u64_of(&r["window_tokens"]) {
                    0 => "–".into(),
                    w => thousands(w),
                },
                bytes(u64_of(&r["stride_bytes"])),
            ];
            match regions.iter_mut().find(|(known, _)| *known == row) {
                Some((_, count)) => *count += 1,
                None => regions.push((row, 1)),
            }
        }
        if !regions.is_empty() {
            let rows: Vec<Vec<String>> = regions
                .into_iter()
                .map(|(mut row, count)| {
                    row.push(count.to_string());
                    row
                })
                .collect();
            canvas.table(
                &[
                    "history region",
                    "tokens/record",
                    "window",
                    "record stride",
                    "count",
                ],
                &rows,
            );
        }
        canvas.close();

        canvas.open("SPECULATION");
        let widths = list(&speculation["widths"]);
        if widths.is_empty() && u64_of(&speculation["token_capacity"]) == 0 {
            canvas.note("no speculative graphs prepared; every step decodes one token per request");
        } else {
            canvas.row(
                "proposal source",
                match text(&speculation["proposal_source"]).as_str() {
                    "mtp" => "multi-token prediction head",
                    "copy" => "copy drafts from shared decoded history",
                    other => other,
                },
            );
            canvas.number("verification widths", &widths);
            canvas.number(
                "fixed width",
                &match speculation["fixed_width"].as_u64() {
                    Some(w) => format!("{w} (adaptive verification off)"),
                    None => "none (adaptive verification)".into(),
                },
            );
            canvas.number(
                "token capacity",
                &format!(
                    "{} target rows per speculative step",
                    thousands(u64_of(&speculation["token_capacity"]))
                ),
            );
            canvas.row(
                "history drafts",
                if speculation["history_drafts"].as_bool().unwrap_or(false) {
                    "yes"
                } else {
                    "no"
                },
            );
        }
        canvas.close();
    }
    if value.get("frontend").is_some() {
        canvas.open("FRONTEND");
        canvas.row(
            "prompt encoder",
            match text(&frontend["prompt_encoder"]).as_str() {
                "PROMPT_JINJA" => "Jinja chat template",
                "DEEPSEEK_V4" => "DeepSeek V4 native encoder",
                "DEEPSEEK_V41" => "DeepSeek V4.1 native encoder",
                other => other,
            },
        );
        canvas.row(
            "template encoding",
            match text(&frontend["template_encoding"]).as_str() {
                "TEMPLATE_TEXT" => "text",
                "TEMPLATE_TOKEN_SEGMENTS" => "token segments",
                other => other,
            },
        );
        let tokenizer = &frontend["tokenizer"];
        canvas.number(
            "eos tokens",
            &format!(
                "{}{}",
                list(&frontend["eos_token_ids"]),
                match text(&tokenizer["eos_token"]).as_str() {
                    "" => String::new(),
                    token => format!(" · {token}"),
                }
            ),
        );
        for (key, name) in [("bos_token", "bos token"), ("pad_token", "pad token")] {
            let token = text(&tokenizer[key]);
            if !token.is_empty() {
                canvas.row(name, &token);
            }
        }
        canvas.number(
            "special tokens",
            &format!(
                "{} additional",
                u64_of(&tokenizer["additional_special_tokens"])
            ),
        );
        match frontend["tools"].as_object() {
            Some(tools) => canvas.row(
                "tool calls",
                &format!(
                    "{} parser{}",
                    text(&tools["kind"]).to_lowercase().replace('_', " "),
                    match u64_of(&tools["wrapping_newlines"]) {
                        0 => String::new(),
                        n => format!(" · {n} wrapping newlines"),
                    }
                ),
            ),
            None => canvas.row("tool calls", "no parser"),
        }
        match frontend["reasoning"].as_object() {
            Some(reasoning) => canvas.row(
                "reasoning",
                &format!(
                    "{} parser · {} … {}",
                    text(&reasoning["kind"]).to_lowercase().replace('_', " "),
                    text(&reasoning["open"]),
                    text(&reasoning["close"])
                ),
            ),
            None => canvas.row("reasoning", "no parser"),
        }
        let marker = text(&frontend["assistant_marker"]);
        if !marker.is_empty() {
            canvas.row("assistant marker", &marker);
        }
        let ends = list(&frontend["message_end_markers"]);
        if !ends.is_empty() {
            canvas.row("message end markers", &ends);
        }
        if let Some(boundary) = frontend["prefix_boundary"].as_object() {
            let roles = list(&boundary["roles"]);
            if !roles.is_empty() {
                canvas.row(
                    "prefix boundary",
                    &format!(
                        "{roles}{}",
                        if boundary["continue_final_message"]
                            .as_bool()
                            .unwrap_or(false)
                        {
                            " · continues the final message"
                        } else {
                            ""
                        }
                    ),
                );
            }
        }
        let assets: Vec<Vec<String>> = value["assets"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|a| {
                vec![
                    text(&a["name"]),
                    bytes(u64_of(&a["bytes"])),
                    text(&a["sha256"]).chars().take(16).collect(),
                ]
            })
            .collect();
        if !assets.is_empty() {
            canvas.table(&["asset", "size", "sha256"], &assets);
        }
        canvas.close();

        if let Some(vision) = vision.as_object() {
            canvas.open("VISION");
            let image = &vision["image"];
            canvas.row("image profile", &text(&image["profile"]));
            canvas.number(
                "image tokens per prompt",
                &vision["max_image_tokens_per_prompt"].as_u64()
                    .map(|limit| format!("{} (shared context)", thousands(limit)))
                    .unwrap_or_else(|| "not declared".into()),
            );
            canvas.number(
                "images per prompt",
                &vision["max_images"].as_u64().map(thousands)
                    .unwrap_or_else(|| "not declared".into()),
            );
            canvas.number(
                "image feature window",
                &thousands(u64_of(&vision["image_feature_window_tokens"])),
            );
            canvas.number(
                "patches per token",
                &u64_of(&vision["patches_per_feature"]).to_string(),
            );
            let capacities = vision["patch_capacities"]
                .as_array()
                .map(|c| {
                    c.iter()
                        .map(|v| thousands(u64_of(v)))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            canvas.number("vision graph patches", &capacities);
            canvas.number(
                "feature slots",
                &u64_of(&vision["feature_slots"]).to_string(),
            );
            for (key, name) in [
                ("min_pixels", "min pixels"),
                ("max_pixels", "max pixels"),
                ("max_tokens", "max tokens"),
            ] {
                if let Some(n) = image[key].as_u64() {
                    canvas.number(name, &thousands(n));
                }
            }
            if let Some(token) = frontend["image_placeholder_token_id"].as_u64() {
                canvas.number("placeholder token", &token.to_string());
            }
            canvas.close();
        }

        if let Some(engram) = engram.as_object() {
            canvas.open("ENGRAM");
            canvas.number(
                "compressed vocabulary",
                &thousands(u64_of(&engram["compressed_vocabulary"])),
            );
            canvas.number(
                "token map",
                &format!(
                    "{} entries",
                    thousands(u64_of(&engram["token_map_entries"]))
                ),
            );
            let tables: Vec<Vec<String>> = engram["tables"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|t| {
                    vec![
                        u64_of(&t["layer"]).to_string(),
                        thousands(u64_of(&t["rows"])),
                        u64_of(&t["embedding_width"]).to_string(),
                        text(&t["encoding"]).to_lowercase(),
                        bytes(u64_of(&t["weight_bytes"])),
                        bytes(u64_of(&t["scale_bytes"])),
                        u64_of(&t["shards"]).to_string(),
                    ]
                })
                .collect();
            canvas.table(
                &[
                    "layer", "rows", "width", "encoding", "weights", "scales", "shards",
                ],
                &tables,
            );
            canvas.close();
        }
    }

    for rank in &ranks {
        if let Some(memory) = rank.get("memory_map") {
            memory_map(&mut canvas, rank, memory);
        }
        if let Some(graphs) = rank["execution_graphs"].as_array() {
            canvas.open(&format!("Graphs · rank {}", rank["rank"]));
            canvas.table(
                &[
                    "Graph", "Kind", "Batch", "Tokens", "Packed", "Nodes", "Edges", "Time",
                ],
                &graphs
                    .iter()
                    .map(|g| {
                        vec![
                            text(&g["name"]),
                            text(&g["kind"]),
                            text(&g["batch_capacity"]),
                            text(&g["token_capacity"]),
                            text(&g["packed_tokens"]),
                            text(&g["node_count"]),
                            text(&g["edge_count"]),
                            duration(u64_of(&g["estimated_duration_ns"])),
                        ]
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
    canvas.close();
    canvas.out.trim_end().to_string() + "\n"
}

fn sampling(canvas: &mut Canvas, sampling: &Value) {
    canvas.open("Sampling");
    let mut rows = Vec::new();
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    ] {
        let value = &sampling[key];
        let default = if value.is_f64() {
            decimal(f64_of(value))
        } else if value.is_null() {
            "not recorded".into()
        } else {
            text(value)
        };
        let range = &sampling["ranges"][key];
        let bounds = if key == "seed" {
            "request controlled".into()
        } else if range.is_object() {
            format!(
                "{} – {}",
                decimal(f64_of(&range["min"])),
                decimal(f64_of(&range["max"]))
            )
        } else {
            "fixed".into()
        };
        rows.push(vec![key.replace('_', " "), default, bounds]);
    }
    canvas.table(&["Parameter", "Default", "Request range"], &rows);
}

fn memory_map(canvas: &mut Canvas, rank: &Value, memory: &Value) {
    canvas.open(&format!("Memory map · rank {}", rank["rank"]));
    canvas.note("Recorded virtual addresses and extents; runtime caches are backed at load time.");
    for a in memory["allocations"].as_array().into_iter().flatten() {
        canvas.row(
            &text(&a["name"]),
            &format!(
                "{} · {} · {} virtual · {} stored",
                a["address"]
                    .as_u64()
                    .map_or_else(|| "relocatable".into(), |n| format!("0x{n:016x}")),
                text(&a["memory"]),
                bytes(u64_of(&a["virtual_bytes"])),
                bytes(u64_of(&a["stored_bytes"]))
            ),
        );
        if a["sizing"].is_object() {
            canvas.row(
                "  capacity",
                &format!(
                    "{} × {} bytes; max {} units; alignment {} bytes",
                    text(&a["sizing"]["capacity"]),
                    a["sizing"]["unit_bytes"],
                    a["sizing"]["maximum"],
                    a["alignment"]
                ),
            );
        }
        if a["payload"].is_object() {
            canvas.row(
                "  payload",
                &format!(
                    "{} + {} ({} bytes)",
                    if a["payload"]["filename"] == "" {
                        "primary".into()
                    } else {
                        text(&a["payload"]["filename"])
                    },
                    a["payload"]["offset"],
                    a["payload"]["nbytes"]
                ),
            );
        }
    }
    canvas.open(&format!("Address reservations · rank {}", rank["rank"]));
    canvas.table(
        &["Address", "Bytes"],
        &memory["reservations"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|r| {
                vec![
                    format!("0x{:016x}", u64_of(&r["address"])),
                    text(&r["nbytes"]),
                ]
            })
            .collect::<Vec<_>>(),
    );
    canvas.open(&format!("Execution buffers · rank {}", rank["rank"]));
    for b in memory["buffers"].as_array().into_iter().flatten() {
        canvas.row(
            &text(&b["name"]),
            &format!(
                "{} {} · {} bytes · {} + {}",
                text(&b["dtype"]),
                text(&b["shape"]),
                b["nbytes"],
                text(&b["reference"]["allocation"]),
                b["reference"]["offset"]
            ),
        );
        canvas.row("  strides", &format!("{} elements", text(&b["strides"])));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "package": {
                "path": "/packages/model.safetensors", "format_version": 5, "bytes": 700_000_000u64,
                "companions": [{"name": "model-data-01-of-02.safetensors", "bytes": 60_000_000_000u64},
                               {"name": "model-data-02-of-02.safetensors", "bytes": 50_000_000_000u64}],
                "ranks": 1, "cluster": null,
                "publisher": {"name": "Local development signer", "namespace": "local-development",
                              "fingerprint": "ab".repeat(32), "development": true},
                "official_source": "https://huggingface.co/org/model",
                "license": {"name": "MIT", "link": "https://example.com/LICENSE"},
            },
            "model": {
                "model_id": "org/Model-Flash", "identity": "cd".repeat(32), "served_names": ["Ava"],
                "vocabulary": 151936, "input_vocabulary": 151936, "max_sequence_tokens": 1048576,
                "prefill_chunk_size": 4096, "max_active": 32, "max_new_tokens": 32768,
                "decode_weight_bytes": 0, "reserve_bytes": 2147483648u64, "queue_capacity": 16,
                "frontend_workers": 2, "max_body_bytes": 268435456, "context_hint": "", "capacity_hint": "",
            },
            "cache": {"page_tokens": 64, "pages": 64, "state_rows": 64, "snapshots": 32,
                      "regions": [{"kind": "PAGED_LATENT", "tokens_per_record": 64, "window_tokens": 0,
                                   "stride_bytes": 36864, "nbytes": 1},
                                  {"kind": "PAGED_LATENT", "tokens_per_record": 64, "window_tokens": 0,
                                   "stride_bytes": 36864, "nbytes": 1},
                                  {"kind": "SLIDING_WINDOW", "tokens_per_record": 1, "window_tokens": 4096,
                                   "stride_bytes": 512, "nbytes": 1}]},
            "speculation": {"proposal_source": "mtp", "widths": [5, 3], "fixed_width": null,
                            "token_capacity": 192, "history_drafts": false},
            "sampling_defaults": {"seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                                  "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
                                  "frequency_penalty": 0.0},
            "frontend": {"prompt_encoder": "DEEPSEEK_V41", "template_encoding": "TEMPLATE_TEXT",
                         "eos_token_ids": [1], "image_placeholder_token_id": 128815,
                         "tokenizer": {"eos_token": "<｜end▁of▁sentence｜>", "bos_token": "<｜begin▁of▁sentence｜>",
                                       "pad_token": null, "unk_token": null, "additional_special_tokens": 12},
                         "tools": {"kind": "DSML", "wrapping_newlines": 1, "tag_separator": ""},
                         "reasoning": {"kind": "THINK_TAGS", "open": "<think>", "close": "</think>",
                                       "wrapping_newlines": 0, "terminal_markers": []},
                         "assistant_marker": "<|im_start|>assistant\n", "message_end_markers": [],
                         "rejected_tokens": 0, "position_dimensions": 1, "prefix_boundary": null},
            "vision": {"feature_slots": 4, "max_image_tokens_per_prompt": 1048576, "max_images": 1048576,
                       "image_feature_window_tokens": 4096,
                       "patches_per_feature": 9, "patch_capacities": [3456], "image": {"profile": "deepseek4"}},
            "assets": [{"name": "tokenizer", "bytes": 11_000_000, "sha256": "ef".repeat(32)}],
            "engram": {"compressed_vocabulary": 120000, "pad_token_id": 2, "token_map_entries": 151936,
                       "tables": [{"layer": 3, "rows": 4194304, "embedding_width": 512, "encoding": "NVFP4_E4M3_GLOBAL",
                                   "weight_bytes": 1_073_741_824u64, "scale_bytes": 134_217_728u64, "shards": 0}]},
            "ranks": [{"rank": 0, "architecture": [10, 3], "device_bytes": 160_000_000_000u64,
                       "mapped_host_bytes": 1_000_000, "peer_bytes": 0, "allocations": 40,
                       "capacity_allocations": [{"name": "kv", "capacity": "KV_PAGES", "unit_bytes": 36864,
                                                 "maximum": 57344, "memory": "DEVICE"},
                                                {"name": "state", "capacity": "STATE_ROWS", "unit_bytes": 4096,
                                                 "maximum": 64, "memory": "DEVICE"}],
                       "modules": 12, "functions": 3000, "graphs": 24, "events": 2, "abi_version": 9, "max_active": 32,
                       "execution_graphs": [
                           {"name": "decode_b32", "kind": "decode", "sampling_mode": "sample", "batch_capacity": 32,
                            "token_capacity": 1, "packed_tokens": 32, "prefill_capacity": 0, "draft_tokens": null,
                            "verify_tokens": null, "draft_source": "NO_DRAFT", "speculation_enabled": false,
                            "estimated_duration_ns": 2_100_000},
                           {"name": "verification_w5_b32_with_a_very_long_name_that_will_not_fit_anywhere", "kind": "verification",
                            "sampling_mode": "sample", "batch_capacity": 32, "token_capacity": 6, "packed_tokens": 192,
                            "prefill_capacity": 0, "draft_tokens": 5, "verify_tokens": 6, "draft_source": "MTP",
                            "speculation_enabled": true, "estimated_duration_ns": 4_000}],
                       "roce": null}],
            "graph_contracts": [],
        })
    }

    fn strip_escapes(text: &str) -> String {
        let mut output = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                output.push(c);
            }
        }
        output
    }

    #[test]
    fn summary_is_compact_and_details_follow_document_selection() {
        let mut summary = document();
        summary["package"]["signatures_verified"] = true.into();
        summary["cache"]["history_kinds"] = json!(["PAGED_LATENT", "SLIDING_WINDOW"]);
        for key in ["frontend", "vision", "engram", "assets"] {
            summary.as_object_mut().unwrap().remove(key);
        }
        summary["ranks"][0]
            .as_object_mut()
            .unwrap()
            .remove("execution_graphs");
        let plain = render(&summary, 100, false);
        assert!(plain.lines().count() < 50, "{plain}");
        assert!(plain.contains("signatures verified"), "{plain}");
        assert!(!plain.contains("/packages/model.safetensors"));
        let model = plain
            .split("╭─ Model ")
            .nth(1)
            .unwrap()
            .split('╰')
            .next()
            .unwrap();
        assert!(model.contains("publisher"));
        assert!(model.contains("license"));
        assert!(plain.contains("1,048,576 tokens"));
        assert!(plain.contains("Sampling"));
        assert!(!plain.contains("Frontend"));
        assert!(!plain.contains("Graphs · rank"));
        assert!(!plain.chars().any(|c| matches!(c, '░' | '\x1b')));
        assert_eq!(
            plain.lines().filter(|line| line.starts_with("╭─ ")).count(),
            3
        );
        assert_eq!(
            plain.lines().filter(|line| line.starts_with('╰')).count(),
            3
        );
        for line in plain.lines().filter(|line| line.starts_with('│')) {
            assert_eq!(visible(line), 100, "{line:?}");
            assert!(line.ends_with('│'));
        }
        let full = render(&document(), 100, false);
        assert!(full.contains("FRONTEND"));
        assert!(full.contains("Graphs · rank 0"));
        assert!(full.contains("image feature window"));
        assert!(full.contains("shared context"));
    }

    #[test]
    fn model_title_uses_the_served_name_and_fits_narrow_terminals() {
        let mut value = document();
        value["model"]["model_id"] = json!("1bae0e1088993b9b82de08f59de5be5a34e42170");
        value["model"]["served_names"] = json!(["deepseek-v4.1-flash"]);
        for width in [64, 80, 100, 120] {
            let rendered = render(&value, width, false);
            let title = rendered.split("╭─ Model ").next().unwrap();
            assert!(title.starts_with('\n'));
            assert!(!title.contains("deepseek-v4.1-flash"));
            let model = rendered
                .split("╭─ Model ")
                .nth(1)
                .unwrap()
                .split('╰')
                .next()
                .unwrap();
            assert!(
                model
                    .lines()
                    .any(|line| line.starts_with("│   name ")
                        && line.contains("deepseek-v4.1-flash"))
            );
            assert!(!title.contains(value["model"]["model_id"].as_str().unwrap()));
            assert!(title.lines().all(|line| visible(line) <= width));
            assert!(title.chars().any(|c| matches!(c, '█' | '▀' | '▄')));
            let rows = title.lines().filter(|line| line.contains('█')).count();
            assert_eq!(rows, if width < 120 { 8 } else { 4 }, "{title}");
        }
        value["model"]["served_names"] = json!(["abcdefghijklmnopqrstuvwxyz0123456789"]);
        let rendered = render(&value, 64, false);
        assert!(rendered.lines().all(|line| visible(line) <= 64));
        value["model"]["served_names"] = json!(["日本語モデル"]);
        let rendered = render(&value, 80, false);
        assert!(
            !rendered
                .split("╭─ Model ")
                .next()
                .unwrap()
                .contains("日本語モデル")
        );
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with("│   name ") && line.contains("日本語モデル"))
        );
    }

    #[test]
    fn styled_output_preserves_plain_text_and_width() {
        for width in [MIN_WIDTH, 80, MAX_WIDTH] {
            let plain = render(&document(), width, false);
            let styled = render(&document(), width, true);
            assert!(styled.contains("\x1b[1;36m"));
            assert_eq!(strip_escapes(&styled), plain);
            for line in plain.lines() {
                assert!(visible(line) <= width, "{width}: {line:?}");
            }
        }
    }

    #[test]
    fn sampling_displays_declared_bounds_without_invented_ranges() {
        let mut value = document();
        value["sampling_defaults"]["ranges"] = json!({"temperature":{"min":0.7,"max":1.0}});
        let rendered = render(&value, 100, false);
        assert!(rendered.contains("0.7 – 1"), "{rendered}");
        assert!(rendered.contains("fixed"));
        assert!(!rendered.contains("0 – 2"));
        assert!(rendered.contains("request controlled"));
    }

    #[test]
    fn formatting_helpers() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2 KiB");
        assert_eq!(bytes(3 * 1024 * 1024), "3.0 MiB");
        assert_eq!(bytes(5 * 1024 * 1024 * 1024), "5.00 GiB");
        assert_eq!(decimal(1.0), "1");
        assert_eq!(decimal(0.95), "0.95");
        assert_eq!(decimal(0.0), "0");
        assert_eq!(duration(0), "–");
        assert_eq!(Canvas::wrap("abc def ghi", 7), vec!["abc def", "ghi"]);
        assert_eq!(Canvas::wrap("abcdefghijkl", 8), vec!["abcdefgh", "ijkl"]);
    }
}
