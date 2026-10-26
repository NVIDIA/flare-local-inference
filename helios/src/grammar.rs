//! Native JSON-schema and llguidance Lark constraints over exact tokenizer bytes.

use std::sync::Arc;

use llguidance::api::{GrammarWithLexer, ParserLimits, TopLevelGrammar};
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory, token_bytes_from_tokenizer_json};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{AdmissionError, Error, Failure, Result};

const MAX_GRAMMAR_BYTES: usize = 262_144;
const MAX_SCHEMA_NODES: usize = 8_192;
const MAX_SCHEMA_DEPTH: usize = 64;

/// Delimiters supplied by the same protocol that parses generated output.
pub struct OutputEnvelope {
    pub reasoning_end: Option<&'static str>,
    pub tool: Option<(&'static str, &'static str)>,
    pub function: Option<(&'static str, &'static str)>,
    pub tool_ends_reasoning: bool,
    pub tool_body_start: bool,
    pub response: Option<(&'static str, &'static str)>,
}

struct CheckpointTokenizer {
    tokenizer: Tokenizer,
    trie: TokTrie,
}

impl TokenizerEnv for CheckpointTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.tokenizer
                .encode(text, false)
                .expect("validated byte-level checkpoint tokenizer failed to encode UTF-8")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_is_canonical(&self) -> bool {
        // NFC normalization need not preserve grammar bytes. Masking uses the exact
        // byte trie; disabling token forcing prevents retokenization of those bytes.
        false
    }
}

/// Shared tokenizer and bounded grammar compiler; each request owns its parser.
pub struct GrammarCompiler {
    factory: ParserFactory,
    model_vocab: usize,
    tokenizer_vocab: usize,
    eos_ids: Arc<[u32]>,
    delimiters: Arc<[(u32, String)]>,
}

impl GrammarCompiler {
    /// `delimiters` are single-token output delimiters that grammars reference by
    /// token ID, so their text spelled in ordinary pieces never matches them. Their
    /// bytes remain ordinary text, which answers may contain.
    pub fn new(
        tokenizer: &Tokenizer,
        model_vocab: usize,
        eos_ids: &[u32],
        delimiters: &[(u32, String)],
    ) -> Result<Self> {
        if model_vocab == 0 || model_vocab > u32::MAX as usize {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar model vocabulary must fit a nonzero u32",
            ));
        }
        let json = tokenizer.to_string(false).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        let json: Value = serde_json::from_str(&json).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::caused_by(
                error.to_string(),
                error,
            )))
        })?;
        let has_byte_decoder = json["decoder"]["type"] == "ByteLevel"
            || (json["decoder"]["type"] == "Sequence"
                && json["decoder"]["decoders"]
                    .as_array()
                    .is_some_and(|decoders| decoders.iter().any(|d| d["type"] == "ByteFallback")));
        if !has_byte_decoder {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar requires a ByteLevel or ByteFallback checkpoint decoder",
            ));
        }
        let bytes = token_bytes_from_tokenizer_json(&json).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::boxed(
                error.to_string(),
                error.into_boxed_dyn_error(),
            )))
        })?;
        if bytes.len() > model_vocab || bytes.is_empty() || bytes.iter().any(Vec::is_empty) {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar tokenizer has missing IDs or exceeds the model vocabulary",
            ));
        }
        if delimiters.iter().any(|&(id, _)| id as usize >= bytes.len()) {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar delimiter token exceeds the tokenizer vocabulary",
            ));
        }
        if eos_ids.is_empty()
            || eos_ids.iter().enumerate().any(|(index, &id)| {
                id as usize >= bytes.len()
                    || eos_ids[..index].contains(&id)
                    || bytes[id as usize][0] != TokTrie::SPECIAL_TOKEN_MARKER
            })
        {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar EOS IDs must be distinct checkpoint special tokens",
            ));
        }
        let tokenizer_vocab = bytes.len();
        let trie = TokTrie::from(&TokRxInfo::new(tokenizer_vocab as u32, eos_ids[0]), &bytes)
            .with_eos_tokens(eos_ids);
        let mut tokenizer = tokenizer.clone();
        tokenizer.with_truncation(None).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        let environment: TokEnv = Arc::new(CheckpointTokenizer { tokenizer, trie });
        let mut factory = ParserFactory::new_simple(&environment).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::boxed(
                error.to_string(),
                error.into_boxed_dyn_error(),
            )))
        })?;
        factory.quiet();
        *factory.limits_mut() = ParserLimits {
            max_grammar_size: 131_072,
            max_lexer_states: 32_768,
            verbose_errors: false,
            ..ParserLimits::default()
        };
        Ok(Self {
            factory,
            model_vocab,
            tokenizer_vocab,
            eos_ids: eos_ids.into(),
            delimiters: delimiters.into(),
        })
    }

    /// The delimiters these grammars reference as tokens, which output parsers gate on.
    pub fn delimiters(&self) -> &[(u32, String)] {
        &self.delimiters
    }

    /// Compile supported JSON-schema assertions strictly; external references and
    /// compiler option extensions are rejected before compilation.
    pub fn compile_json_schema(&self, schema: &Value) -> Result<RequestGrammar> {
        self.compile_json_schema_in(schema, None)
    }

    pub(crate) fn compile_json_schema_in(
        &self,
        schema: &Value,
        envelope: Option<&OutputEnvelope>,
    ) -> Result<RequestGrammar> {
        if schema.to_string().len() > MAX_GRAMMAR_BYTES {
            return Err(Error::admission(
                AdmissionError::Grammar,
                format!("JSON schema exceeds {MAX_GRAMMAR_BYTES} bytes"),
            ));
        }
        validate_schema_nodes(schema, "$", 0, &mut 0)?;
        self.compile_envelope(TopLevelGrammar::from_json_schema(schema.clone()), envelope)
    }

    /// Compile the llguidance Lark dialect. Embedded JSON and compiler directives
    /// use separate APIs so they cannot override strict JSON-schema validation.
    pub fn compile_lark(&self, grammar: &str) -> Result<RequestGrammar> {
        self.compile_lark_in(grammar, None)
    }

    pub(crate) fn compile_lark_in(
        &self,
        grammar: &str,
        envelope: Option<&OutputEnvelope>,
    ) -> Result<RequestGrammar> {
        validate_lark(grammar)?;
        self.compile_envelope(TopLevelGrammar::from_lark(grammar.to_owned()), envelope)
    }

    fn compile_envelope(
        &self,
        mut grammar: TopLevelGrammar,
        envelope: Option<&OutputEnvelope>,
    ) -> Result<RequestGrammar> {
        let Some(envelope) = envelope else {
            return self.compile(grammar);
        };
        if envelope.reasoning_end.is_none()
            && envelope.tool.is_none()
            && envelope.response.is_none()
            && !envelope.tool_body_start
        {
            return self.compile(grammar);
        }
        let mut rules = String::from("ws: /[ \\t\\r\\n]*/\n");
        if let Some((open, close)) = envelope.response {
            rules.push_str(&format!(
                "wrapped_answer: {} @answer {}\nempty_answer: {} {}\n",
                self.marker(open),
                self.marker(close),
                self.marker(open),
                self.marker(close)
            ));
        } else {
            rules.push_str("wrapped_answer: @answer\n");
        }
        let branches = if let Some((open, close)) = envelope.tool {
            if let Some((function_open, function_close)) = envelope.function {
                rules.push_str(&format!(
                    "tool_after_open: ws {} tool_body (ws {})?\ntool: {} tool_after_open | tool_after_open\n",
                    self.marker(function_open), self.marker(close), self.marker(open)
                ));
                rules.push_str(&self.region("tool_body", function_close));
            } else {
                rules.push_str(&format!(
                    "tool_after_open: tool_body\ntool: {} tool_after_open\n",
                    self.marker(open)
                ));
                rules.push_str(&self.region("tool_body", close));
            }
            rules.push_str(
                "tools_suffix: (ws tool)* (ws wrapped_answer)?\ntools: tool tools_suffix\n",
            );
            if envelope.response.is_some() {
                "(wrapped_answer (ws tool)* | empty_answer ws tools)"
            } else {
                "(wrapped_answer | tools)"
            }
        } else {
            "wrapped_answer"
        };
        if envelope.tool_body_start {
            if envelope.tool.is_none() {
                return Err("tool-body output start requires tool syntax".into());
            }
            rules.push_str("start: tool_after_open tools_suffix ws\n");
        } else if let Some(end) = envelope.reasoning_end {
            if let Some((open, _)) = envelope.tool.filter(|_| envelope.tool_ends_reasoning) {
                rules.push_str(&self.delimited_region("reasoning_done", end, &[end, open]));
                rules.push_str(&self.delimited_region("reasoning_tool", open, &[end, open]));
                // Only the outer tool opener implicitly ends reasoning.
                rules.push_str(&format!(
                    "start: (reasoning_done ws {branches} | reasoning_tool tool_after_open tools_suffix) ws\n",
                ));
            } else {
                rules.push_str(&self.region("reasoning", end));
                rules.push_str(&format!("start: reasoning ws {branches} ws\n"));
            }
        } else {
            rules.push_str(&format!("start: ws {branches} ws\n"));
        }
        grammar.grammars[0].name = Some("answer".into());
        grammar
            .grammars
            .insert(0, GrammarWithLexer::from_lark(rules));
        self.compile(grammar)
    }

    fn marker(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        match self.token_id(text) {
            Some(id) => format!("<[{id}]>"),
            None => serde_json::to_string(text).unwrap(),
        }
    }

    /// The ID of a declared delimiter or special token, which grammars match only as
    /// that token.
    fn token_id(&self, text: &str) -> Option<u32> {
        self.delimiters
            .iter()
            .find(|(_, marker)| marker == text)
            .map(|&(id, _)| id)
            .or_else(|| self.factory.tok_env().tok_trie().get_special_token(text))
    }

    fn region(&self, name: &str, close: &str) -> String {
        let trie = self.factory.tok_env().tok_trie();
        if let Some(end) = self.token_id(close) {
            let specials: Vec<_> = trie
                .get_special_tokens()
                .into_iter()
                .filter(|id| *id != end && !self.eos_ids.contains(id))
                .map(|id| id.to_string())
                .collect();
            let body = if specials.is_empty() {
                "/(?s:.*)/".into()
            } else {
                format!("(/(?s:.+)/ | <[{}]>)*", specials.join(","))
            };
            format!("{name}: {body} <[{end}]>\n")
        } else {
            // Lazy suffixes consume the delimiter without token backtracking.
            format!(
                "{name}[suffix={}]: /(?s:.*)/\n",
                serde_json::to_string(close).unwrap()
            )
        }
    }

    fn delimited_region(&self, name: &str, close: &str, ends: &[&str]) -> String {
        let trie = self.factory.tok_env().tok_trie();
        let literals: Vec<_> = ends
            .iter()
            .filter(|end| self.token_id(end).is_none())
            .map(|end| serde_json::to_string(end).unwrap())
            .collect();
        let text = if literals.is_empty() {
            "/(?s:.*)/".to_owned()
        } else {
            format!(
                "(/(?s:.*)/ & ~(/(?s:.*)/ ({}) /(?s:.*)/))",
                literals.join(" | ")
            )
        };
        let excluded: Vec<_> = ends.iter().filter_map(|end| self.token_id(end)).collect();
        let specials: Vec<_> = trie
            .get_special_tokens()
            .into_iter()
            .filter(|id| !excluded.contains(id) && !self.eos_ids.contains(id))
            .map(|id| id.to_string())
            .collect();
        let terminal = format!("{}_TEXT", name.to_uppercase());
        let ending = if let Some(id) = self.token_id(close) {
            format!("{name}_end: {terminal} <[{id}]>\n")
        } else {
            format!(
                "{name}_end[suffix={}]: {terminal}\n",
                serde_json::to_string(close).unwrap()
            )
        };
        let prefix = if specials.is_empty() {
            String::new()
        } else {
            format!("({terminal} <[{}]>)* ", specials.join(","))
        };
        format!("{terminal}: {text}\n{ending}{name}: {prefix}{name}_end\n")
    }

    /// Compile a complete custom-tool input contract without sampling masks.
    pub(crate) fn compile_text(&self, syntax: &str, source: &str) -> Result<TextGrammar> {
        let grammar = match syntax {
            "lark" => {
                validate_lark(source)?;
                TopLevelGrammar::from_lark(source.to_owned())
            }
            "regex" if !source.is_empty() && source.len() <= MAX_GRAMMAR_BYTES => {
                TopLevelGrammar::from_regex(source)
            }
            _ => {
                return Err(Error::admission(
                    AdmissionError::Grammar,
                    "custom tool format requires a bounded Lark or regex grammar",
                ));
            }
        };
        let parser = self.factory.create_parser(grammar).map_err(|e| {
            Error::admission(
                AdmissionError::Grammar,
                format!("invalid custom tool grammar: {e:#}"),
            )
        })?;
        let mut matcher = Matcher::new(Ok(parser));
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() || matcher.is_error() {
            return Err(Error::admission(
                AdmissionError::Grammar,
                format!(
                    "invalid custom tool grammar: {}",
                    matcher.get_error().unwrap_or_else(|| warnings.join("; "))
                ),
            ));
        }
        Ok(TextGrammar { matcher })
    }

    fn compile(&self, grammar: TopLevelGrammar) -> Result<RequestGrammar> {
        let parser = self.factory.create_parser(grammar).map_err(|error| {
            Error::Admission(AdmissionError::Grammar(Failure::boxed(
                format!("invalid grammar: {error:#}"),
                error.into_boxed_dyn_error(),
            )))
        })?;
        let mut matcher = Matcher::new(Ok(parser));
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(Error::admission(
                AdmissionError::Grammar,
                format!(
                    "grammar compilation warnings are rejected: {}",
                    warnings.join("; ")
                ),
            ));
        }
        let mut request = RequestGrammar {
            matcher,
            mask: vec![0; self.model_vocab.div_ceil(32)],
            tokenizer_vocab: self.tokenizer_vocab,
            eos_ids: self.eos_ids.clone(),
            accepting: false,
            stopped: false,
            error: None,
        };
        request.refresh_mask()?;
        Ok(request)
    }
}

fn validate_lark(grammar: &str) -> Result<()> {
    if grammar.is_empty() || grammar.len() > MAX_GRAMMAR_BYTES {
        return Err(Error::admission(
            AdmissionError::Grammar,
            format!("Lark grammar must contain 1..={MAX_GRAMMAR_BYTES} bytes"),
        ));
    }
    if grammar.contains("%json") || grammar.contains("%llguidance") {
        return Err(Error::admission(
            AdmissionError::Grammar,
            "Lark %json and %llguidance directives are unsupported; use the JSON-schema API for schemas",
        ));
    }
    Ok(())
}

pub(crate) struct TextGrammar {
    matcher: Matcher,
}
impl TextGrammar {
    pub fn validate(&self, text: &str) -> Result<()> {
        let mut matcher = self.matcher.clone();
        let environment = matcher
            .tok_env()
            .map_err(|e| format!("custom tool grammar: {e}"))?;
        let trie = environment.tok_trie();
        // Greedy tokenization preserves the exact bytes, without checkpoint normalization.
        let tokens = trie.greedy_tokenize(text.as_bytes());
        if trie.decode_raw(&tokens) != text.as_bytes() {
            return Err(
                "custom tool input cannot be represented by the tokenizer byte trie".into(),
            );
        }
        let consumed = matcher
            .try_consume_tokens(&tokens)
            .map_err(|e| format!("custom tool input failed grammar validation: {e}"))?;
        if consumed != tokens.len() || !matcher.is_accepting().map_err(|e| e.to_string())? {
            return Err("custom tool input does not match its grammar".into());
        }
        Ok(())
    }
}

/// Per-generation parser. Advance once per sampled completion before uploading
/// the following mask; prompt tokens and consumed pending tokens do not advance it.
pub struct RequestGrammar {
    matcher: Matcher,
    mask: Vec<i32>,
    tokenizer_vocab: usize,
    eos_ids: Arc<[u32]>,
    accepting: bool,
    stopped: bool,
    error: Option<Error>,
}

impl RequestGrammar {
    /// Returns masks before each legal non-EOS draft and after the final draft.
    /// The committed parser, including its lexer caches, remains independent.
    pub fn copy_masks(&self, tokens: &[u32]) -> Result<(usize, Vec<i32>)> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.stopped {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "cannot propose tokens for a stopped grammar",
            ));
        }
        if tokens.len() > 7 {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "copy grammar supports at most seven draft tokens",
            ));
        }
        let mut tentative = Self {
            matcher: self.matcher.deep_clone(),
            mask: self.mask.clone(),
            tokenizer_vocab: self.tokenizer_vocab,
            eos_ids: self.eos_ids.clone(),
            accepting: self.accepting,
            stopped: false,
            error: None,
        };
        let mut masks = Vec::with_capacity((tokens.len() + 1) * self.mask.len());
        masks.extend_from_slice(&self.mask);
        let mut count = 0;
        for &token in tokens {
            let id = token as usize;
            // EOS remains a target outcome; a stopped parser has no bonus mask.
            if self.eos_ids.contains(&token)
                || id >= self.tokenizer_vocab
                || (tentative.mask[id / 32] as u32 & (1 << (id % 32))) == 0
            {
                break;
            }
            if let Err(error) = tentative.matcher.consume_token(token) {
                return Err(Error::Admission(AdmissionError::Grammar(Failure::boxed(
                    format!("grammar advancement failed: {error:#}"),
                    error.into_boxed_dyn_error(),
                ))));
            }
            if !tentative.refresh_mask_available()? {
                break;
            }
            masks.extend_from_slice(tentative.mask());
            count += 1;
        }
        Ok((count, masks))
    }

    /// Int32 words with token t stored in bit (t % 32) of word (t / 32).
    /// Padded model IDs and the mask of a stopped or failed parser are zero.
    pub fn mask(&self) -> &[i32] {
        &self.mask
    }

    pub fn is_accepting(&self) -> bool {
        self.accepting && self.error.is_none()
    }

    /// True after a permitted EOS token was consumed, not merely at a valid prefix.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub fn advance(&mut self, token: u32) -> Result<()> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.stopped {
            return Err(Error::admission(
                AdmissionError::Grammar,
                "grammar already consumed EOS",
            ));
        }
        let id = token as usize;
        if id >= self.tokenizer_vocab || (self.mask[id / 32] as u32 & (1 << (id % 32))) == 0 {
            return self.fail(Error::admission(
                AdmissionError::Grammar,
                format!("token {token} is forbidden by the grammar mask"),
            ));
        }
        if self.eos_ids.contains(&token) {
            if !self.accepting {
                return self.fail(Error::admission(
                    AdmissionError::Grammar,
                    "grammar EOS reached a non-accepting parser state",
                ));
            }
            self.stopped = true;
            self.mask.fill(0);
            return Ok(());
        }
        if let Err(error) = self.matcher.consume_token(token) {
            return self.fail(Error::Admission(AdmissionError::Grammar(Failure::boxed(
                format!("grammar advancement failed: {error:#}"),
                error.into_boxed_dyn_error(),
            ))));
        }
        self.refresh_mask()
    }

    fn refresh_mask(&mut self) -> Result<()> {
        if !self.refresh_mask_available()? {
            return self.fail(Error::admission(
                AdmissionError::Grammar,
                "grammar has no allowed checkpoint tokens",
            ));
        }
        Ok(())
    }

    fn refresh_mask_available(&mut self) -> Result<bool> {
        self.mask.fill(0);
        if !self.matcher.stop_reason().is_ok() {
            return self.fail(Error::admission(
                AdmissionError::Grammar,
                format!("grammar stopped with {:?}", self.matcher.stop_reason()),
            ));
        }
        self.accepting = match self.matcher.is_accepting() {
            Ok(value) => value,
            Err(error) => {
                return self.fail(Error::Admission(AdmissionError::Grammar(Failure::boxed(
                    format!("grammar acceptance check failed: {error:#}"),
                    error.into_boxed_dyn_error(),
                ))));
            }
        };
        let allowed = match self.matcher.compute_mask_or_eos() {
            Ok(mask) => mask,
            Err(error) => {
                return self.fail(Error::Admission(AdmissionError::Grammar(Failure::boxed(
                    format!("grammar mask computation failed: {error:#}"),
                    error.into_boxed_dyn_error(),
                ))));
            }
        };
        for (destination, &word) in self.mask.iter_mut().zip(allowed.as_slice()) {
            *destination = word as i32;
        }
        let tail = self.tokenizer_vocab % 32;
        if tail != 0 {
            self.mask[self.tokenizer_vocab / 32] &= ((1_u32 << tail) - 1) as i32;
        }
        Ok(self.mask.iter().any(|&word| word != 0))
    }

    fn fail<T>(&mut self, error: Error) -> Result<T> {
        self.mask.fill(0);
        self.accepting = false;
        self.error = Some(error.clone());
        Err(error)
    }
}

fn validate_schema_nodes(
    schema: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<()> {
    *nodes += 1;
    if depth > MAX_SCHEMA_DEPTH || *nodes > MAX_SCHEMA_NODES {
        return Err(Error::admission(
            AdmissionError::Grammar,
            "JSON schema exceeds the depth or node limit",
        ));
    }
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema.as_object().ok_or_else(|| {
        Error::admission(
            AdmissionError::Grammar,
            format!("schema {path} must be an object or boolean"),
        )
    })?;
    for (key, value) in object {
        let child = format!("{path}/{key}");
        match key.as_str() {
            "$ref" => {
                if !value
                    .as_str()
                    .is_some_and(|reference| reference.starts_with('#'))
                {
                    return Err(Error::admission(
                        AdmissionError::Grammar,
                        format!("schema {child} must reference the same document"),
                    ));
                }
            }
            "$defs" | "definitions" | "properties" | "patternProperties" => {
                let entries = value.as_object().ok_or_else(|| {
                    Error::admission(
                        AdmissionError::Grammar,
                        format!("schema {child} must be an object"),
                    )
                })?;
                for (name, nested) in entries {
                    validate_schema_nodes(nested, &format!("{child}/{name}"), depth + 1, nodes)?;
                }
            }
            "anyOf" | "oneOf" | "allOf" | "prefixItems" => {
                let entries = value.as_array().ok_or_else(|| {
                    Error::admission(
                        AdmissionError::Grammar,
                        format!("schema {child} must be an array"),
                    )
                })?;
                for (index, nested) in entries.iter().enumerate() {
                    validate_schema_nodes(nested, &format!("{child}/{index}"), depth + 1, nodes)?;
                }
            }
            "items" if value.is_array() => {
                for (index, nested) in value.as_array().unwrap().iter().enumerate() {
                    validate_schema_nodes(nested, &format!("{child}/{index}"), depth + 1, nodes)?;
                }
            }
            "items" | "additionalItems" | "additionalProperties" => {
                validate_schema_nodes(value, &child, depth + 1, nodes)?;
            }
            "$anchor" | "$schema" | "$id" | "id" | "$comment" | "title" | "description"
            | "default" | "readOnly" | "writeOnly" | "examples" | "contentMediaType"
            | "contentEncoding" | "const" | "enum" | "type" | "minItems" | "maxItems"
            | "required" | "minProperties" | "maxProperties" | "minLength" | "maxLength"
            | "pattern" | "format" | "minimum" | "maximum" | "exclusiveMinimum"
            | "exclusiveMaximum" | "multipleOf" => {}
            _ => {
                return Err(Error::admission(
                    AdmissionError::Grammar,
                    format!("unsupported JSON-schema keyword {child}"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn byte_tokenizer() -> Tokenizer {
        byte_tokenizer_with(&[])
    }

    /// Adds non-special tokens from 258, as checkpoints declare reasoning delimiters.
    fn byte_tokenizer_with(added: &[&str]) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        let mut extra = 256;
        for byte in 0..=255_u32 {
            let mapped = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                byte
            } else {
                extra += 1;
                extra - 1
            };
            vocab.insert(char::from_u32(mapped).unwrap().to_string(), json!(byte));
        }
        let mut added_tokens = vec![
            json!({"id":256,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}),
            json!({"id":257,"content":"<end>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}),
        ];
        for (id, content) in (258..).zip(added) {
            added_tokens.push(json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
        }
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "post_processor": null,
            "decoder": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":vocab,"merges":[]}
        });
        Tokenizer::from_bytes(tokenizer.to_string()).unwrap()
    }

    /// SentencePiece-style vocabulary: `<unk>` is 0, `<0xNN>` byte tokens are 1..=256.
    fn byte_fallback_tokenizer(decoders: Value) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".to_string(), json!(0));
        for byte in 0..=255_u32 {
            vocab.insert(format!("<0x{byte:02X}>"), json!(byte + 1));
        }
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id":257,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":258,"content":"<end>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type":"Sequence","decoders":decoders},
            "model": {"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,"vocab":vocab,"merges":[]}
        });
        Tokenizer::from_bytes(tokenizer.to_string()).unwrap()
    }

    fn allowed(request: &RequestGrammar, token: u32) -> bool {
        request.mask()[token as usize / 32] as u32 & (1 << (token % 32)) != 0
    }

    fn accepts(compiler: &GrammarCompiler, schema: &Value, text: &str, eos: u32) -> bool {
        let mut request = compiler.compile_json_schema(schema).unwrap();
        for &token in text.as_bytes() {
            if request.advance(u32::from(token)).is_err() {
                return false;
            }
        }
        request.advance(eos).is_ok()
    }

    #[test]
    fn byte_fallback_decoder_compiles_byte_token_grammars() {
        let tokenizer = byte_fallback_tokenizer(json!([
            {"type":"Replace","pattern":{"String":"\u{2581}"},"content":" "},
            {"type":"ByteFallback"},
            {"type":"Fuse"}
        ]));
        let compiler = GrammarCompiler::new(&tokenizer, 320, &[257, 258], &[]).unwrap();
        let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
        let advance_text = |text: &str| {
            let mut request = compiler.compile_json_schema(&schema).unwrap();
            for &byte in text.as_bytes() {
                if request.advance(u32::from(byte) + 1).is_err() {
                    return false;
                }
            }
            request.advance(258).is_ok()
        };
        assert!(advance_text(r#"{"ok":true}"#));
        assert!(!advance_text(r#"{"ok":1}"#));
    }

    #[test]
    fn decoder_without_byte_tokens_is_rejected() {
        let tokenizer = byte_fallback_tokenizer(json!([{"type":"Fuse"}]));
        assert!(GrammarCompiler::new(&tokenizer, 320, &[257, 258], &[]).is_err());
    }

    #[test]
    fn exact_mask_words_eos_and_padded_ids() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut request = compiler.compile_lark(r"start: /[\x1f]/").unwrap();
        assert_eq!(request.mask().len(), 10);
        assert_eq!(request.mask()[0], i32::MIN);
        assert_eq!(request.mask().iter().filter(|&&word| word != 0).count(), 1);
        request.advance(31).unwrap();
        assert!(request.is_accepting());
        assert!(!request.is_stopped());
        assert!(allowed(&request, 256) && allowed(&request, 257));
        assert!((258..320).all(|id| !allowed(&request, id)));
        request.advance(257).unwrap();
        assert!(request.is_stopped());
        assert!(request.mask().iter().all(|&word| word == 0));
        assert!(request.advance(257).is_err());
    }

    #[test]
    fn output_envelope_constrains_only_the_answer() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"const":{"ok":true}});
        for reasoning in [false, true] {
            let envelope = OutputEnvelope {
                reasoning_end: reasoning.then_some("</think>"),
                tool: Some(("<tool_call>", "</tool_call>")),
                function: None,
                tool_ends_reasoning: false,
                response: None,
                tool_body_start: false,
            };
            for (text, expected) in [
                (r#"{"ok":true}"#, true),
                (r#"{"ok":false}"#, false),
                ("<tool_call>arbitrary arguments</tool_call>", true),
                (
                    "<tool_call>one</tool_call>\n<tool_call>two</tool_call>",
                    true,
                ),
                ("<tool_call>one</tool_call>\n{\"ok\":true}", true),
                ("<tool_call>one</tool_call>\n{\"ok\":false}", false),
                ("<tool_call>one</tool_call>junk", false),
                ("<tool_call>unterminated", false),
                ("plain text", false),
                ("", false),
            ] {
                let text = if reasoning {
                    format!("Think about {{nonsense}} and <tool_call> literally.</think>\n\n{text}")
                } else {
                    text.into()
                };
                let mut grammar = compiler
                    .compile_json_schema_in(&schema, Some(&envelope))
                    .unwrap();
                let accepted = text
                    .bytes()
                    .map(u32::from)
                    .chain([256])
                    .all(|token| grammar.advance(token).is_ok());
                assert_eq!(accepted, expected, "{text}");
            }
        }
    }

    #[test]
    fn xtml_response_envelope_constrains_answers_and_allows_tool_calls() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        for reasoning in [false, true] {
            let envelope = OutputEnvelope {
                reasoning_end: reasoning.then_some("<|close|>think<|sep|><|open|>response<|sep|>"),
                tool: Some(("<|open|>tools<|sep|>", "<|close|>tools<|sep|>")),
                function: None,
                tool_ends_reasoning: false,
                response: Some(("", "<|close|>response<|sep|>")),
                tool_body_start: false,
            };
            for (body, expected) in [
                (r#"{"ok":true}<|close|>response<|sep|>"#, true),
                (r#"{"ok":false}<|close|>response<|sep|>"#, false),
                (
                    "<|close|>response<|sep|><|open|>tools<|sep|>call<|close|>tools<|sep|>",
                    true,
                ),
                ("<|open|>tools<|sep|>call<|close|>tools<|sep|>", false),
                (r#"{"ok":true}"#, false),
            ] {
                let text = if reasoning {
                    format!("Consider.{}{}", envelope.reasoning_end.unwrap(), body)
                } else {
                    body.into()
                };
                let mut grammar = compiler
                    .compile_json_schema_in(&json!({"const":{"ok":true}}), Some(&envelope))
                    .unwrap();
                let accepted = text
                    .bytes()
                    .map(u32::from)
                    .chain([256])
                    .all(|token| grammar.advance(token).is_ok());
                assert_eq!(accepted, expected, "{text}");
            }
        }
    }

    #[test]
    fn qwen_envelope_preserves_implicit_reasoning_and_function_boundaries() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let envelope = OutputEnvelope {
            reasoning_end: Some("</think>"),
            tool: Some(("<tool_call>", "</tool_call>")),
            function: Some(("<function=", "</function>")),
            tool_ends_reasoning: true,
            response: None,
            tool_body_start: false,
        };
        for (text, accepted) in [
            ("thought</think>{\"ok\":true}", true),
            ("thought<tool_call><function=f></function>", true),
            (
                "thought<tool_call><function=f></function></tool_call>",
                true,
            ),
            ("thought</think><function=f></function>{\"ok\":true}", true),
            (
                "thought<tool_call><function=f></function><function=f></function>",
                true,
            ),
            (
                "thought<tool_call>literal</tool_call></think>{\"ok\":true}",
                false,
            ),
            ("thought</think>{\"ok\":false}", false),
            (
                "thought<tool_call><function=f></function>junk</tool_call>",
                false,
            ),
        ] {
            let mut grammar = compiler
                .compile_json_schema_in(&json!({"const":{"ok":true}}), Some(&envelope))
                .unwrap();
            let result = text
                .bytes()
                .map(u32::from)
                .chain([256])
                .all(|id| grammar.advance(id).is_ok());
            assert_eq!(result, accepted, "{text}");
        }
    }

    #[test]
    fn single_token_reasoning_delimiters_end_reasoning_only_as_tokens() {
        let tokenizer = byte_tokenizer_with(&["<think>", "</think>"]);
        let delimiters = [(258, "<think>".into()), (259, "</think>".into())];
        let compiler = GrammarCompiler::new(&tokenizer, 320, &[256, 257], &delimiters).unwrap();
        // \x01 and \x02 stand for the <think> and </think> tokens; other bytes are text.
        let ids = |text: &str| -> Vec<u32> {
            text.bytes()
                .map(|byte| match byte {
                    1 => 258,
                    2 => 259,
                    byte => u32::from(byte),
                })
                .collect()
        };
        for tool_ends_reasoning in [false, true] {
            let envelope = OutputEnvelope {
                reasoning_end: Some("</think>"),
                tool: Some(("<tool_call>", "</tool_call>")),
                function: Some(("<function=", "</function>")),
                tool_ends_reasoning,
                response: None,
                tool_body_start: false,
            };
            for (text, accepted) in [
                ("Spell </think> and <think> here.\x02{\"ok\":true}", true),
                ("Spelled </think>{\"ok\":false}\x02{\"ok\":true}", true),
                ("plan\x01more\x02{\"ok\":true}", true),
                ("plan\x02{\"ok\":false}", false),
                ("plan</think>{\"ok\":true}", false),
            ] {
                let mut grammar = compiler
                    .compile_json_schema_in(&json!({"const":{"ok":true}}), Some(&envelope))
                    .unwrap();
                let result = ids(text)
                    .into_iter()
                    .chain([256])
                    .all(|id| grammar.advance(id).is_ok());
                assert_eq!(result, accepted, "{text:?} {tool_ends_reasoning}");
            }
            for (text, accepted) in [
                ("plan\x02\"\x01a\x02 </think>\"", true),
                ("plan\x02\"a\"\x02", false),
            ] {
                let mut grammar = compiler
                    .compile_json_schema_in(&json!({"type":"string"}), Some(&envelope))
                    .unwrap();
                let result = ids(text)
                    .into_iter()
                    .chain([256])
                    .all(|id| grammar.advance(id).is_ok());
                assert_eq!(result, accepted, "{text:?} {tool_ends_reasoning}");
            }
        }
    }

    #[test]
    fn output_envelope_copy_masks_cross_reasoning_and_tool_boundaries() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let envelope = OutputEnvelope {
            reasoning_end: Some("</think>"),
            tool: Some(("<tool_call>", "</tool_call>")),
            function: Some(("<function=", "</function>")),
            tool_ends_reasoning: true,
            response: None,
            tool_body_start: false,
        };
        let schema = json!({"const":7});
        let text = "thought<tool_call><function=f></function></tool_call>7";
        let tokens: Vec<_> = text.bytes().map(u32::from).collect();
        let mut grammar = compiler
            .compile_json_schema_in(&schema, Some(&envelope))
            .unwrap();
        for start in 0..tokens.len() {
            let end = (start + 7).min(tokens.len());
            let mask = grammar.mask().to_vec();
            let (count, masks) = grammar.copy_masks(&tokens[start..end]).unwrap();
            assert_eq!(count, end - start);
            assert_eq!(grammar.mask(), mask);
            assert_eq!(&masks[..mask.len()], mask);
            grammar.advance(tokens[start]).unwrap();
            assert_eq!(&masks[mask.len()..2 * mask.len()], grammar.mask());
        }
        grammar.advance(256).unwrap();
    }

    #[test]
    #[ignore = "requires FLARE_TEST_SNAPSHOT with the Qwen tokenizer"]
    fn qwen_output_envelope_uses_checkpoint_token_bytes() {
        let path = std::path::PathBuf::from(std::env::var_os("FLARE_TEST_SNAPSHOT").unwrap());
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).unwrap();
        let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
        let think = tokenizer.token_to_id("<think>").unwrap();
        let end = tokenizer.token_to_id("</think>").unwrap();
        let delimiters = [(think, "<think>".into()), (end, "</think>".into())];
        let compiler = GrammarCompiler::new(&tokenizer, 248320, &[eos], &delimiters).unwrap();
        let encode = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        // Spells reasoning delimiters in ordinary pieces.
        let spelled = |text: &str| -> Vec<u32> {
            let mut ids = Vec::new();
            let mut rest = text;
            while let Some(at) = ["<think>", "</think>"]
                .iter()
                .filter_map(|delimiter| rest.find(delimiter))
                .min()
            {
                ids.extend(encode(&rest[..at]));
                ids.extend(encode("<"));
                rest = &rest[at + 1..];
            }
            ids.extend(encode(rest));
            ids
        };
        let per_char = |text: &str| -> Vec<u32> {
            text.chars()
                .flat_map(|ch| encode(&ch.to_string()))
                .collect()
        };
        let schema = json!({"type":"object","properties":{"literal":{"type":"string"}},"required":["literal"],"additionalProperties":false});
        let literal = r#"{"literal":"<think>東京</think><tool_call>literal</tool_call>"}"#;
        for reasoning in [false, true] {
            let envelope = OutputEnvelope {
                reasoning_end: reasoning.then_some("</think>"),
                tool: Some(("<tool_call>", "</tool_call>")),
                function: Some(("<function=", "</function>")),
                tool_ends_reasoning: true,
                response: None,
                tool_body_start: false,
            };
            let prefix = |pieces: &dyn Fn(&str) -> Vec<u32>| -> Vec<u32> {
                if reasoning {
                    [
                        pieces("Consider </think> first."),
                        vec![end],
                        pieces("\n\n"),
                    ]
                    .concat()
                } else {
                    Vec::new()
                }
            };
            let mut grammar = compiler
                .compile_json_schema_in(&schema, Some(&envelope))
                .unwrap();
            for token in [prefix(&spelled), encode(r#"{"literal":""#)].concat() {
                grammar.advance(token).unwrap();
            }
            assert!(allowed(&grammar, think) && allowed(&grammar, end));
            for text in [
                literal,
                "<tool_call><function=lookup><parameter=query>東京</parameter></function></tool_call>",
            ] {
                let canonical = [prefix(&spelled), spelled(text)].concat();
                let fragmented = [prefix(&per_char), per_char(text)].concat();
                let delimiter_tokens = [prefix(&spelled), encode(text)].concat();
                for tokens in [canonical, fragmented, delimiter_tokens] {
                    let mut grammar = compiler
                        .compile_json_schema_in(&schema, Some(&envelope))
                        .unwrap();
                    for (at, &token) in tokens.iter().enumerate() {
                        let (count, _) = grammar
                            .copy_masks(&tokens[at..(at + 6).min(tokens.len())])
                            .unwrap();
                        assert_eq!(count, (tokens.len() - at).min(6), "at {at} in {text}");
                        grammar.advance(token).unwrap();
                    }
                    grammar.advance(eos).unwrap();
                    assert!(grammar.is_stopped());
                }
            }
        }
    }

    #[test]
    fn custom_tool_grammars_validate_complete_exact_text_repeatedly() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        for (syntax, source) in [
            ("regex", "café[0-9]+"),
            ("lark", "start: \"café\" /[0-9]+/"),
        ] {
            let grammar = compiler.compile_text(syntax, source).unwrap();
            for _ in 0..2 {
                assert!(grammar.validate("café42").is_ok());
                for text in ["", "café", "cafe\u{301}42", "café42tail", "prefixcafé42"] {
                    assert!(grammar.validate(text).is_err(), "{syntax}: {text:?}");
                }
            }
        }
        assert!(
            compiler
                .compile_text("lark", "start: missing_rule")
                .is_err()
        );
        assert!(
            compiler
                .compile_text("lark", "%llguidance {}\nstart: /./")
                .is_err()
        );
        assert!(compiler.compile_text("regex", "(").is_err());
    }

    #[test]
    fn required_fields_enums_ranges_and_additional_properties() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"type":"object","properties":{
            "action":{"enum":["read","write"]},
            "count":{"type":"integer","minimum":1,"maximum":3}
        },"required":["action","count"],"additionalProperties":false});
        for text in [
            r#"{"action":"read","count":1}"#,
            r#"{"action":"write", "count":3}"#,
        ] {
            assert!(accepts(&compiler, &schema, text, 256), "{text}");
        }
        for text in [
            r#"{"action":"read"}"#,
            r#"{"action":"erase","count":1}"#,
            r#"{"action":"read","count":0}"#,
            r#"{"action":"read","count":4}"#,
            r#"{"action":"read","count":2,"extra":true}"#,
            r#"{"action":"read","count":"2"}"#,
        ] {
            assert!(!accepts(&compiler, &schema, text, 257), "{text}");
        }
    }

    #[test]
    fn local_references_arrays_unicode_and_nullable_values() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let schema = json!({"$defs":{"item":{"anyOf":[{"type":"null"},{"enum":["é","猫","😀"]}]}},
            "type":"array","items":{"$ref":"#/$defs/item"},"minItems":1,"maxItems":2});
        for text in [r#"[null]"#, r#"["é","猫"]"#, r#"["😀"]"#] {
            assert!(accepts(&compiler, &schema, text, 257), "{text}");
        }
        for text in ["[]", "[null,null,null]", "[false]", r#"["wrong"]"#] {
            assert!(!accepts(&compiler, &schema, text, 256), "{text}");
        }
    }

    #[test]
    fn parser_failure_preserves_typed_first_cause() {
        use std::error::Error as _;

        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut request = compiler.compile_lark("start: \"a\"").unwrap();
        let original = request
            .fail::<()>(Error::Admission(AdmissionError::Grammar(
                Failure::caused_by(
                    "parser fixture failure",
                    std::io::Error::from_raw_os_error(5),
                ),
            )))
            .unwrap_err();
        for cached in [
            request.advance(b'a' as u32).unwrap_err(),
            request.copy_masks(&[]).unwrap_err(),
        ] {
            assert!(matches!(
                cached,
                Error::Admission(AdmissionError::Grammar(_))
            ));
            assert_eq!(cached.to_string(), "parser fixture failure");
            assert!(std::ptr::eq(
                cached.source().unwrap(),
                original.source().unwrap()
            ));
            assert_eq!(
                cached
                    .source()
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .unwrap()
                    .raw_os_error(),
                Some(5)
            );
        }
        assert!(request.mask().iter().all(|&word| word == 0));
    }

    #[test]
    fn compiler_failure_retains_library_cause() {
        use std::error::Error as _;

        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let error = compiler.compile_lark("start: missing_rule").err().unwrap();
        assert!(matches!(
            error,
            Error::Admission(AdmissionError::Grammar(_))
        ));
        assert!(error.source().is_some());
        let invalid = compiler.compile_lark("").err().unwrap();
        assert!(matches!(
            invalid,
            Error::Admission(AdmissionError::Grammar(_))
        ));
        assert!(invalid.source().is_none());
    }

    #[test]
    fn masked_tokens_poison_parser_without_permissive_fallback() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut request = compiler
            .compile_json_schema(&json!({"const":true}))
            .unwrap();
        assert!(!allowed(&request, 256));
        let error = request.advance(b'f' as u32).unwrap_err();
        assert!(matches!(
            error,
            Error::Admission(AdmissionError::Grammar(_))
        ));
        assert_eq!(
            error.to_string(),
            "token 102 is forbidden by the grammar mask"
        );
        let cached = request.advance(b't' as u32).unwrap_err();
        assert!(matches!(
            cached,
            Error::Admission(AdmissionError::Grammar(_))
        ));
        assert_eq!(cached.to_string(), error.to_string());
        assert!(request.mask().iter().all(|&word| word == 0));
        assert!(!request.is_accepting());
    }

    #[test]
    fn copy_masks_include_bonus_and_rejection_does_not_advance_committed_parser() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut committed = compiler.compile_lark("start: \"abc\" | \"axy\"").unwrap();
        committed.advance(b'a' as u32).unwrap();
        let before = committed.mask().to_vec();
        let (count, masks) = committed
            .copy_masks(&[b'b' as u32, b'c' as u32, 256])
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(masks.len(), 3 * before.len());
        assert_eq!(committed.mask(), before);
        let mut serial = compiler.compile_lark("start: \"abc\" | \"axy\"").unwrap();
        serial.advance(b'a' as u32).unwrap();
        for (index, mask) in masks.chunks(before.len()).enumerate() {
            assert_eq!(mask, serial.mask());
            if index < 2 {
                serial.advance(b"bc"[index] as u32).unwrap();
            }
        }
        assert!(serial.is_accepting());
        assert!(masks[2 * before.len() + 256 / 32] as u32 & 1 != 0);
        // A correction at the first candidate follows the unmodified parser.
        committed.advance(b'x' as u32).unwrap();
        committed.advance(b'y' as u32).unwrap();
        committed.advance(257).unwrap();
        assert!(committed.is_stopped());
        assert!(committed.copy_masks(&[]).is_err());
    }

    #[test]
    fn copy_masks_trim_forbidden_ids_without_poisoning_and_preserve_utf8_prefixes() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        let mut committed = compiler
            .compile_json_schema(&json!({"const":"猫"}))
            .unwrap();
        let original = committed.mask().to_vec();
        let tokens = "\"猫\""
            .as_bytes()
            .iter()
            .map(|b| *b as u32)
            .collect::<Vec<_>>();
        let (count, masks) = committed.copy_masks(&tokens).unwrap();
        assert_eq!(count, tokens.len());
        assert_eq!(committed.mask(), original);
        for (index, &token) in tokens.iter().enumerate() {
            assert_eq!(&masks[index * 10..(index + 1) * 10], committed.mask());
            committed.advance(token).unwrap();
        }
        assert_eq!(&masks[count * 10..], committed.mask());
        assert_eq!(
            committed.copy_masks(&[u32::MAX]).unwrap(),
            (0, committed.mask().to_vec())
        );
        assert!(committed.advance(256).is_ok());
        let parser = compiler.compile_lark("start: \"ab\"").unwrap();
        let (count, masks) = parser.copy_masks(&[b'a' as u32, b'x' as u32]).unwrap();
        assert_eq!((count, masks.len()), (1, 20));
        assert_eq!(
            parser.copy_masks(&[b'x' as u32]).unwrap(),
            (0, parser.mask().to_vec())
        );
    }

    #[test]
    fn unsupported_invalid_and_external_schemas_fail_before_admission() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        for schema in [
            json!({"type":"array","uniqueItems":true}),
            json!({"type":"integer","minimim":1}),
            json!({"properties":{"x":{"nullable":true}}}),
            json!({"type":"string","minLength":-1}),
            json!({"type":"string","format":"not-a-format"}),
            json!({"oneOf":[{"type":"integer"},{"type":"number"}]}),
            json!({"x-guidance":{"lenient":true},"uniqueItems":true}),
            json!({"$ref":"https://example.invalid/schema"}),
            json!({"$ref":"file:///etc/passwd"}),
            json!({"$ref":"#/$defs/missing"}),
            json!({"type":"integer","minimum":5,"maximum":1}),
            json!(false),
        ] {
            assert!(
                compiler.compile_json_schema(&schema).is_err(),
                "accepted {schema}"
            );
        }
        assert!(
            compiler
                .compile_lark("start: %json {\"x-guidance\":{\"lenient\":true}}")
                .is_err()
        );
        assert!(compiler.compile_lark("start: missing").is_err());
    }

    #[test]
    fn bounded_compilation_and_thread_ownership() {
        fn send_sync<T: Send + Sync>() {}
        fn send<T: Send>() {}
        send_sync::<GrammarCompiler>();
        send::<RequestGrammar>();
        let compiler =
            Arc::new(GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap());
        let mut schema = json!(true);
        for _ in 0..=MAX_SCHEMA_DEPTH {
            schema = json!({"type":"array","items":schema});
        }
        assert!(compiler.compile_json_schema(&schema).is_err());
        assert!(
            compiler
                .compile_lark(&" ".repeat(MAX_GRAMMAR_BYTES + 1))
                .is_err()
        );
        let compiler2 = compiler.clone();
        let mut request =
            std::thread::spawn(move || compiler2.compile_json_schema(&json!({"const":7})).unwrap())
                .join()
                .unwrap();
        request.advance(b'7' as u32).unwrap();
        request.advance(256).unwrap();
        assert!(request.is_stopped());
    }

    #[test]
    fn reject_tokenizer_or_eos_geometry_mismatch() {
        let tokenizer = byte_tokenizer();
        assert!(GrammarCompiler::new(&tokenizer, 257, &[256], &[]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[], &[]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[256, 256], &[]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[256, 319], &[]).is_err());
        assert!(GrammarCompiler::new(&tokenizer, 320, &[b'a' as u32], &[]).is_err());
    }

    #[test]
    #[ignore = "requires FLARE_CHECKPOINT pointing to the exact Qwen3.8 checkpoint"]
    fn checkpoint_mask_trace() {
        let snapshot = std::path::PathBuf::from(
            std::env::var_os("FLARE_CHECKPOINT").expect("FLARE_CHECKPOINT is required"),
        );
        let reference = check_checkpoint_masks(&snapshot, None);
        let trace_path = std::env::temp_dir().join(format!(
            "flare-grammar-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let traced = check_checkpoint_masks(&snapshot, Some(&trace_path));
        let saved: Value = serde_json::from_slice(&std::fs::read(&trace_path).unwrap()).unwrap();
        std::fs::remove_file(trace_path).unwrap();
        assert_eq!(reference, traced);
        assert_eq!(reference, saved);
    }

    fn check_checkpoint_masks(
        snapshot: &std::path::Path,
        trace_output: Option<&std::path::Path>,
    ) -> Value {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(snapshot.join("tokenizer.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
        );
        let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
        let compiler = GrammarCompiler::new(&tokenizer, 248_320, &[248_046, 248_044], &[]).unwrap();
        assert_eq!(compiler.tokenizer_vocab, 248_077);
        let cases = [
            (
                "tool_arguments",
                json!({"type":"object","properties":{"path":{"const":"/tmp/猫.txt"},"lines":{"type":"integer","minimum":1,"maximum":3}},"required":["path","lines"],"additionalProperties":false}),
                r#"{"path":"/tmp/猫.txt","lines":2}"#,
            ),
            (
                "unicode_choices",
                json!({"type":"array","items":{"enum":["é","猫","😀"]},"minItems":1,"maxItems":3}),
                r#"["é","猫","😀"]"#,
            ),
        ];
        let mut traces = Vec::new();
        for (name, schema, text) in cases {
            for eos in [248_046, 248_044] {
                let mut request = compiler.compile_json_schema(&schema).unwrap();
                let tokens = tokenizer.encode(text, false).unwrap();
                let mut steps = Vec::new();
                for &token in tokens.get_ids().iter().chain([eos].iter()) {
                    assert!(allowed(&request, token), "{name}: forbidden token {token}");
                    assert!((248_077..248_320).all(|id| !allowed(&request, id)));
                    let words: Vec<u8> = request
                        .mask()
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    steps.push(json!({"token":token,"allowed_count":request.mask().iter().map(|word| word.count_ones()).sum::<u32>(),"mask_sha256":format!("{:x}",Sha256::digest(&words)),"accepting":request.is_accepting()}));
                    request.advance(token).unwrap();
                }
                assert!(request.is_stopped());
                assert_eq!(tokenizer.decode(tokens.get_ids(), false).unwrap(), text);
                traces
                    .push(json!({"name":name,"schema":schema,"text":text,"eos":eos,"steps":steps}));
            }
        }
        let tokenizer_json: Value = serde_json::from_slice(&bytes).unwrap();
        let token_bytes = token_bytes_from_tokenizer_json(&tokenizer_json).unwrap();
        for text in ["é猫😀", "e\u{301}猫😀"] {
            let grammar = format!("start: {}", serde_json::to_string(text).unwrap());
            let tokens: Vec<u32> = text
                .as_bytes()
                .iter()
                .map(|&byte| {
                    token_bytes
                        .iter()
                        .position(|value| value.as_slice() == [byte])
                        .unwrap() as u32
                })
                .chain([248_046])
                .collect();
            let mut request = compiler.compile_lark(&grammar).unwrap();
            let mut consumed = 0;
            let mut steps = Vec::new();
            for &token in &tokens {
                let remaining = &text.as_bytes()[consumed..];
                for id in 0..248_320 {
                    let expected = if id == 248_046 || id == 248_044 {
                        remaining.is_empty()
                    } else {
                        token_bytes
                            .get(id)
                            .is_some_and(|bytes| !bytes.is_empty() && remaining.starts_with(bytes))
                    };
                    assert_eq!(
                        allowed(&request, id as u32),
                        expected,
                        "literal {text:?}, consumed {consumed}, token {id}"
                    );
                }
                let words: Vec<u8> = request
                    .mask()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                steps.push(json!({"token":token,"allowed_count":request.mask().iter().map(|word| word.count_ones()).sum::<u32>(),"mask_sha256":format!("{:x}",Sha256::digest(&words)),"accepting":request.is_accepting()}));
                request.advance(token).unwrap();
                if token != 248_046 {
                    consumed += token_bytes[token as usize].len();
                }
            }
            assert!(request.is_stopped());
            assert_eq!(
                tokenizer
                    .decode(&tokens[..tokens.len() - 1], false)
                    .unwrap(),
                text
            );
            traces.push(json!({"name":"literal_byte_boundaries","lark":grammar,"text":text,"eos":248046,"full_vocab_mask_equality":true,"steps":steps}));
        }
        let mut incomplete = compiler
            .compile_json_schema(&json!({"const":{"ok":true}}))
            .unwrap();
        assert!(incomplete.advance(248_046).is_err());
        let trace = json!({"model":"local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD","revision":"dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2","llguidance":"1.9.1","tokenizers":"0.23.2","mask_words":7760,"tokenizer_vocab":248077,"model_vocab":248320,"traces":traces});
        if let Some(path) = trace_output {
            std::fs::write(path, serde_json::to_string_pretty(&trace).unwrap() + "\n").unwrap();
        }
        trace
    }

    #[test]
    fn prefixed_dsml_tool_body_preserves_json_and_lark_answer_constraints() {
        let compiler = GrammarCompiler::new(&byte_tokenizer(), 320, &[256, 257], &[]).unwrap();
        for (open, close, separator) in [
            ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", ""),
            ("<｜DSML｜ calls>", "</｜DSML｜ calls>", " "),
        ] {
            let envelope = OutputEnvelope {
                reasoning_end: None,
                tool: Some((open, close)),
                function: None,
                tool_ends_reasoning: false,
                tool_body_start: true,
                response: None,
            };
            let body = format!(
                "\n<｜DSML｜{separator}invoke name=\"lookup\"></｜DSML｜{separator}invoke>{close}"
            );
            for lark in [false, true] {
                for (suffix, expected) in [
                    ("", true),
                    (" true", true),
                    (" false", false),
                    (" junk", false),
                ] {
                    let mut grammar = if lark {
                        compiler
                            .compile_lark_in("start: \"true\"", Some(&envelope))
                            .unwrap()
                    } else {
                        compiler
                            .compile_json_schema_in(&json!({"const": true}), Some(&envelope))
                            .unwrap()
                    };
                    let text = format!("{body}{suffix}");
                    let accepted = text
                        .bytes()
                        .map(u32::from)
                        .chain([256])
                        .all(|token| grammar.advance(token).is_ok());
                    assert_eq!(accepted, expected, "lark={lark}: {text}");
                }
            }
        }
    }
}
