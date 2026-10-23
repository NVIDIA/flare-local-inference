use crate::{AdmissionError, Error, Result};

pub(super) struct TokenStops {
    sequences: Vec<Vec<u32>>,
    held: Vec<u32>,
}
impl TokenStops {
    pub fn new(sequences: Vec<Vec<u32>>, vocabulary: usize) -> Result<Self> {
        if sequences.len() > 16
            || sequences.iter().any(|sequence| {
                sequence.is_empty()
                    || sequence.len() > 64
                    || sequence.iter().any(|token| *token as usize >= vocabulary)
            })
        {
            return Err(Error::admission(
                AdmissionError::Invalid,
                "token stops allow at most sixteen nonempty sequences of at most 64 vocabulary IDs",
            ));
        }
        Ok(Self {
            sequences,
            held: Vec::with_capacity(64),
        })
    }
    pub fn push(&mut self, token: u32) -> (Vec<u32>, bool) {
        self.held.push(token);
        if let Some(length) = self
            .sequences
            .iter()
            .filter(|sequence| self.held.ends_with(sequence))
            .map(Vec::len)
            .max()
        {
            let visible = self.held.drain(..self.held.len() - length).collect();
            self.held.clear();
            return (visible, true);
        }
        let retain = self
            .sequences
            .iter()
            .flat_map(|sequence| {
                (1..sequence.len())
                    .rev()
                    .filter(|length| self.held.ends_with(&sequence[..*length]))
                    .take(1)
            })
            .max()
            .unwrap_or(0);
        (self.held.drain(..self.held.len() - retain).collect(), false)
    }
    pub fn finish(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.held)
    }
}

pub(super) fn validate_strings(strings: &[String], token_stops: bool, grammar: bool) -> Result<()> {
    if strings.len() > 4
        || strings.iter().any(String::is_empty)
        || strings.iter().map(String::len).sum::<usize>() > 4096
    {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "stop allows at most four nonempty strings totaling at most 4096 UTF-8 bytes",
        ));
    }
    if !strings.is_empty() && token_stops {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "stop cannot be combined with stop_token_ids or stop_token_sequences",
        ));
    }
    if !strings.is_empty() && grammar {
        return Err(Error::admission(
            AdmissionError::Invalid,
            "structured output cannot be combined with string stops",
        ));
    }
    Ok(())
}

struct StringStops {
    strings: Vec<String>,
    held: String,
    matched: Option<String>,
}
impl StringStops {
    fn new(strings: Vec<String>) -> Result<Self> {
        validate_strings(&strings, false, false)?;
        Ok(Self {
            strings,
            held: String::new(),
            matched: None,
        })
    }
    fn push(&mut self, text: &str) -> (String, bool) {
        if self.matched.is_some() {
            return (String::new(), true);
        }
        self.held.push_str(text);
        if let Some((_, at, matched)) = self
            .strings
            .iter()
            .enumerate()
            .filter_map(|(i, stop)| self.held.find(stop).map(|at| (at + stop.len(), at, i)))
            .min()
        {
            let visible = self.held[..at].to_owned();
            self.held.clear();
            self.matched = Some(self.strings[matched].clone());
            return (visible, true);
        }
        let retain = self
            .strings
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(length, _)| length)
                    .filter(|length| self.held.ends_with(&stop[..*length]))
            })
            .max()
            .unwrap_or(0);
        (self.held.drain(..self.held.len() - retain).collect(), false)
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

struct Span {
    tokens: Vec<u32>,
    text: String,
    entries: Vec<(usize, super::token_spans::ReturnedLogprob)>,
}

pub(super) struct TextStops<'a> {
    decoder: crate::frontend::IncrementalDecoder<'a>,
    tracked: Option<super::token_spans::TokenSpans<'a>>,
    pending_entries: Vec<(usize, super::token_spans::ReturnedLogprob)>,
    stops: StringStops,
    pending_tokens: Vec<u32>,
    spans: std::collections::VecDeque<Span>,
    visible_bytes: usize,
}
impl<'a> TextStops<'a> {
    pub fn matched(&self) -> Option<&str> {
        self.stops.matched.as_deref()
    }
    pub fn new(frontend: &'a crate::frontend::Frontend, strings: Vec<String>) -> Result<Self> {
        Ok(Self {
            decoder: frontend.incremental_decoder(false),
            tracked: None,
            pending_entries: vec![],
            stops: StringStops::new(strings)?,
            pending_tokens: Vec::new(),
            spans: std::collections::VecDeque::new(),
            visible_bytes: 0,
        })
    }
    pub fn track(&mut self, frontend: &'a crate::frontend::Frontend) {
        self.tracked = Some(super::token_spans::TokenSpans::new(frontend));
    }
    pub fn push_logprob(
        &mut self,
        token: u32,
        value: crate::execution::TokenLogprobs,
        silent: bool,
    ) -> Result<(Vec<crate::output::Piece>, bool)> {
        let tracked = self.tracked.as_mut().ok_or("logprob stop decoder was not initialized")?;
        if silent {
            return Ok(match tracked.silent(token, Some(value)) {
                Some(span) => self.accept_span(span, false),
                None => (vec![], false),
            });
        }
        match tracked.push(token, Some(value))? {
            Some(span) => Ok(self.accept_span(span, false)),
            None => Ok((vec![], false)),
        }
    }
    fn accept_span(&mut self, span: super::token_spans::Span, finish: bool) -> (Vec<crate::output::Piece>, bool) {
        let silent = !span.tokens.is_empty()
            && span.tokens.iter().all(|token| token.logprob.as_ref().is_some_and(|e| e.silent));
        for token in span.tokens {
            self.pending_tokens.push(token.token);
            if let Some(entry) = token.logprob {
                self.pending_entries.push((token.start, entry));
            }
        }
        if silent && span.text.is_empty() {
            self.spans.push_back(Span {
                text: String::new(), tokens: std::mem::take(&mut self.pending_tokens),
                entries: std::mem::take(&mut self.pending_entries),
            });
        }
        self.accept_text(span.text, finish)
    }
    pub fn push(&mut self, token: u32) -> Result<(Vec<crate::output::Piece>, bool)> {
        self.pending_tokens.push(token);
        let text = self.decoder.push(&[token])?;
        Ok(self.accept_text(text.unwrap_or_default(), false))
    }
    pub fn finish(&mut self, eos: Option<u32>) -> Result<(Vec<crate::output::Piece>, bool)> {
        let (mut pieces, matched) = if let Some(tracked) = &mut self.tracked {
            let span = tracked.finish()?;
            self.accept_span(span, true)
        } else {
            let text = self.decoder.finish()?;
            self.accept_text(text, true)
        };
        if !matched {
            pieces.extend(self.pending_entries.drain(..).map(|(_, entry)| crate::output::Piece::Logprob(entry)));
            pieces.extend(
                self.pending_tokens
                    .drain(..)
                    .map(crate::output::Piece::Token),
            );
            pieces.extend(eos.map(crate::output::Piece::Token));
        }
        Ok((pieces, matched))
    }
    fn accept_text(&mut self, text: String, finish: bool) -> (Vec<crate::output::Piece>, bool) {
        use crate::output::Piece;
        let (visible, matched) = self.stops.push(&text);
        self.visible_bytes += visible.len();
        if !text.is_empty() {
            self.spans.push_back(Span {
                tokens: std::mem::take(&mut self.pending_tokens),
                entries: std::mem::take(&mut self.pending_entries),
                text,
            });
        }
        if finish && !matched {
            self.visible_bytes += self.stops.finish().len();
        }
        let mut pieces = Vec::new();
        while self
            .spans
            .front()
            .is_some_and(|span| span.text.len() <= self.visible_bytes)
        {
            let span = self.spans.pop_front().unwrap();
            self.visible_bytes -= span.text.len();
            pieces.extend(span.entries.into_iter().map(|(_, entry)| Piece::Logprob(entry)));
            pieces.extend(span.tokens.into_iter().map(Piece::Token));
            pieces.push(Piece::Text(span.text));
        }
        if matched {
            // A clipped token cannot act as a whole-token protocol control.
            if let Some(span) = self.spans.front() {
                let mut discarded = false;
                for (start, entry) in &span.entries {
                    if *start < self.visible_bytes || (entry.silent && !discarded) {
                        pieces.push(Piece::Logprob(entry.clone()));
                    } else {
                        discarded = true;
                    }
                }
                if self.visible_bytes != 0 {
                    pieces.push(Piece::Text(span.text[..self.visible_bytes].to_owned()));
                }
            }
            self.visible_bytes = 0;
            self.pending_tokens.clear();
            self.pending_entries.clear();
            self.spans.clear();
        }
        (pieces, matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_stop_admission_has_typed_errors() {
        assert!(matches!(
            TokenStops::new(vec![vec![]], 10).err().unwrap(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
        assert!(matches!(
            validate_strings(&[String::new()], false, false).unwrap_err(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
        assert!(matches!(
            validate_strings(&["end".into()], true, false).unwrap_err(),
            Error::Admission(AdmissionError::Invalid(_))
        ));
    }

    #[test]
    fn overlapping_stop_prefixes_are_held_without_losing_ordinary_tokens() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3], vec![2, 4]], 248320).unwrap();
        assert_eq!(stops.push(9), (vec![9], false));
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.push(2), (vec![], false));
        assert_eq!(stops.push(4), (vec![1], true));
        assert!(stops.finish().is_empty());
    }
    #[test]
    fn terminal_flush_preserves_an_unfinished_stop_prefix() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3]], 248320).unwrap();
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.finish(), vec![1]);
    }
}

#[cfg(test)]
pub(super) mod string_tests {
    use super::*;
    use crate::frontend::{Asset, Frontend, FrontendAssets};
    use sha2::{Digest, Sha256};

    pub(crate) fn frontend() -> Frontend {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "flare-stop-tokenizer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = [
            "before", "ENDafter", "<0xE4>", "<0xB8>", "<0x96>", "<0xE7>", "<0x95>", "<0x8C>", "x",
            "ST", "OP", "<0xEF>", "<0xBF>", "<0xBD>",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t.to_owned(), i as u32))
        .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(tokenizers::decoders::sequence::Sequence::new(vec![
            tokenizers::decoders::byte_fallback::ByteFallback::default().into(),
            tokenizers::decoders::fuse::Fuse::default().into(),
        ])));
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let result = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: Some(asset("chat_template.jinja", b"{% mark %}{{ messages[0].content }}")),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        result
    }

    #[test]
    fn first_stop_wins_inside_a_token_and_overlapping_prefixes_are_held() {
        for splits in [vec!["abcSTOPtailEND"], vec!["a", "bcST", "OPtailEND"]] {
            let mut stops =
                StringStops::new(vec!["END".into(), "STOP".into(), "STORK".into()]).unwrap();
            let mut visible = String::new();
            let mut matched = false;
            for piece in splits {
                let (text, hit) = stops.push(piece);
                visible.push_str(&text);
                matched |= hit;
            }
            assert_eq!(visible, "abc");
            assert!(matched);
            assert_eq!(stops.matched.as_deref(), Some("STOP"));
            assert_eq!(stops.push("more"), (String::new(), true));
            assert_eq!(stops.finish(), "");
        }
        let mut stops = StringStops::new(vec!["abc".into(), "bX".into()]).unwrap();
        assert_eq!(stops.push("ab"), (String::new(), false));
        assert_eq!(stops.push("Xafter"), ("a".into(), true));
    }

    #[test]
    fn every_utf8_chunk_boundary_preserves_visible_prefix_and_unmatched_suffix() {
        let full = "café 世界ENDdiscard";
        for split in full.char_indices().map(|(at, _)| at) {
            let mut stops = StringStops::new(vec!["世界END".into(), "é!".into()]).unwrap();
            let (a, _) = stops.push(&full[..split]);
            let (b, hit) = stops.push(&full[split..]);
            assert!(hit);
            assert_eq!(a + &b, "café ");
            assert_eq!(stops.matched.as_deref(), Some("世界END"));
        }
        let mut stops = StringStops::new(vec!["世界END".into()]).unwrap();
        assert_eq!(stops.push("ok世界EN"), ("ok".into(), false));
        assert_eq!(stops.finish(), "世界EN");
    }

    fn text(pieces: Vec<crate::output::Piece>) -> String {
        pieces
            .into_iter()
            .filter_map(|piece| match piece {
                crate::output::Piece::Text(text) => Some(text),
                _ => None,
            })
            .collect()
    }
    fn push_text(stops: &mut TextStops<'_>, token: u32) -> (String, bool) {
        let (pieces, matched) = stops.push(token).unwrap();
        (text(pieces), matched)
    }
    fn finish_text(stops: &mut TextStops<'_>) -> (String, bool) {
        let (pieces, matched) = stops.finish(None).unwrap();
        (text(pieces), matched)
    }

    #[test]
    fn incremental_decoder_handles_split_utf8_and_subtoken_stops() {
        let frontend = frontend();
        let mut stops = TextStops::new(&frontend, vec!["世界".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 0), ("before".into(), false));
        for token in [2, 3, 4, 5, 6] {
            assert_eq!(push_text(&mut stops, token), (String::new(), false));
        }
        assert_eq!(push_text(&mut stops, 7), (String::new(), true));
        assert_eq!(finish_text(&mut stops), (String::new(), true));
        let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 0), ("before".into(), false));
        assert_eq!(push_text(&mut stops, 1), (String::new(), true));
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 9), (String::new(), false));
        assert_eq!(finish_text(&mut stops), ("ST".into(), false));
        let mut stops = TextStops::new(&frontend, vec!["\u{fffd}".into()]).unwrap();
        assert_eq!(push_text(&mut stops, 2), (String::new(), false));
        assert_eq!(finish_text(&mut stops), (String::new(), true));
    }

    #[test]
    fn invalid_or_incompatible_stop_contracts_are_rejected() {
        for values in [
            vec![String::new()],
            vec!["a".into(); 5],
            vec!["x".repeat(4097)],
        ] {
            assert!(validate_strings(&values, false, false).is_err());
        }
        assert!(validate_strings(&["STOP".into()], true, false).is_err());
        assert!(validate_strings(&["STOP".into()], false, true).is_err());
        assert!(validate_strings(&[], true, true).is_ok());
    }
    #[test]
    fn held_spans_release_before_later_token_controls_and_clipped_tokens_have_no_control() {
        use crate::output::Piece::{Text, Token};
        let frontend =
            crate::server::output::tests::frontend(&["beforeST", "<control>", "visibleENDhidden"]);
        let mut stops = TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        assert_eq!(stops.push(0).unwrap(), (vec![], false));
        assert_eq!(
            stops.push(1).unwrap(),
            (
                vec![
                    Token(0),
                    Text("beforeST".into()),
                    Token(1),
                    Text("<control>".into())
                ],
                false
            )
        );
        assert_eq!(stops.finish(Some(99)).unwrap(), (vec![Token(99)], false));
        let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
        assert_eq!(stops.push(2).unwrap(), (vec![Text("visible".into())], true));
        assert_eq!(stops.finish(Some(99)).unwrap(), (vec![], true));
    }
    #[test]
    fn first_completed_overlap_is_independent_of_every_chunk_partition() {
        let input = "abcdef";
        for split_bits in 0..(1 << (input.len() - 1)) {
            let mut stops = StringStops::new(vec!["abcdef".into(), "bc".into()]).unwrap();
            let mut start = 0;
            let mut output = String::new();
            let mut matched = false;
            for end in 1..=input.len() {
                if end == input.len() || split_bits & (1 << (end - 1)) != 0 {
                    let (text, hit) = stops.push(&input[start..end]);
                    output.push_str(&text);
                    matched |= hit;
                    start = end;
                }
            }
            assert!(matched);
            assert_eq!(output, "a", "partition {split_bits}");
        }
        let mut stops = StringStops::new(vec!["bc".into(), "abc".into()]).unwrap();
        assert_eq!(stops.push("abc"), (String::new(), true));
    }
}
