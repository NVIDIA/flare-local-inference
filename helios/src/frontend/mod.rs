//! Native tokenization and rendering over verified Hugging Face assets.

mod deepseek;
mod boundaries;
mod templates;
#[cfg(feature = "images")]
pub mod images;
mod json;
mod jinja;

mod vocabulary;

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{Error, Failure, FrontendError, Result};

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
    pub range: Option<(u64, usize)>,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset path must be absolute: {}",
                    self.path.display()
                ),
            ))));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!("invalid expected SHA256 for {}", self.path.display()),
            ))));
        }
        let bytes = (|| -> std::io::Result<Vec<u8>> {
            if let Some((offset, size)) = self.range {
                use crate::os::FileReadAt;
                let file = fs::File::open(&self.path)?;
                if offset
                    .checked_add(size as u64)
                    .is_none_or(|end| end > file.metadata().map_or(0, |m| m.len()))
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "frontend asset exceeds package",
                    ));
                }
                let mut bytes = vec![0; size];
                file.read_exact_at(&mut bytes, offset)?;
                Ok(bytes)
            } else {
                fs::read(&self.path)
            }
        })()
        .map_err(|error| {
            Error::frontend(FrontendError::Asset(Failure::caused_by(
                format!("cannot read {}: {error}", self.path.display()),
                error,
            )))
        })?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                    self.path.display(),
                    self.sha256
                ),
            ))));
        }
        Ok(bytes)
    }
}

pub struct FrontendAssets {
    pub tokenizer: Asset,
    pub tokens: TokenizerSettings,
    pub chat_template: Option<Asset>,
}

#[cfg(any(test, feature = "test-tools"))]
#[path = "../../tests/support/frontend.rs"]
pub mod test_support;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSettings {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<TokenList>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenList {
    pub values: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendSettings {
    #[serde(default)]
    pub prompt_encoder: PromptEncoder,
    #[serde(default)]
    pub image_placeholder_token_id: Option<u32>,
    pub tokenizer: TokenizerSettings,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub template_encoding: TemplateEncoding,
    #[cfg(feature = "images")]
    pub image: Option<images::ImageSettings>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum PromptEncoder {
    #[default]
    Jinja = 0,
    DeepseekV4 = 1,
    DeepseekV41 = 2,
}

impl From<PromptEncoder> for i32 {
    fn from(value: PromptEncoder) -> Self {
        value as i32
    }
}
impl TryFrom<i32> for PromptEncoder {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Jinja),
            1 => Ok(Self::DeepseekV4),
            2 => Ok(Self::DeepseekV41),
            _ => Err("unsupported prompt encoder"),
        }
    }
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum TemplateEncoding {
    #[default]
    Text = 0,
    TokenSegments = 1,
}

impl From<TemplateEncoding> for i32 {
    fn from(value: TemplateEncoding) -> Self {
        value as i32
    }
}

impl TryFrom<i32> for TemplateEncoding {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Text),
            1 => Ok(Self::TokenSegments),
            _ => Err("unsupported chat template encoding"),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenSegment {
    text: String,
    allow_special: bool,
}

pub struct RenderedPrompt {
    pub image_sources: Vec<String>,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub message_boundaries: Vec<usize>,
}

pub struct Frontend {
    prompt_encoder: PromptEncoder,
    image_placeholder_token_id: Option<u32>,
    tokenizer: Tokenizer,
    ordinary_tokenizer: Option<Tokenizer>,
    template: Option<jinja::CompiledTemplate>,
    template_source: String,
    bos_token_id: Option<u32>,
    effort_levels: std::sync::OnceLock<Vec<String>>,
    vocabulary_bytes: std::sync::OnceLock<Result<vocabulary::VocabularyBytes>>,
    inline_system_messages: std::sync::OnceLock<bool>,
    defaults: Map<String, Json>,
}

impl Frontend {
    pub fn prompt_encoder(&self) -> PromptEncoder {
        self.prompt_encoder
    }
    pub fn with_template_encoding(mut self, encoding: TemplateEncoding) -> Self {
        self.ordinary_tokenizer = match encoding {
            TemplateEncoding::Text => None,
            TemplateEncoding::TokenSegments => {
                let mut tokenizer = self.tokenizer.clone();
                tokenizer.with_added_vocabulary(tokenizers::AddedVocabulary::new());
                Some(tokenizer)
            }
        };
        self
    }
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Resolves advisory effort against the vocabulary accepted by the prepared template.
    pub fn resolve_effort<'a>(&self, candidates: &'a [&str]) -> Result<&'a str> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return candidates
                .iter()
                .copied()
                .find(|level| matches!(*level, "low" | "high" | "xhigh" | "max"))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        let accepted = self.effort_levels.get_or_init(|| {
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([{"role":"user","content":"Hello"}]),
            );
            context.insert("enable_thinking".into(), true.into());
            context.insert("thinking_mode".into(), "thinking".into());
            context.insert("add_generation_prompt".into(), true.into());
            context.insert("continue_final_message".into(), false.into());
            ["low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|effort| {
                    context.insert("reasoning_effort".into(), (*effort).into());
                    self.template
                        .as_ref()
                        .unwrap()
                        .render(&context, &[])
                        .is_ok()
                })
                .map(str::to_owned)
                .collect()
        });
        candidates
            .iter()
            .copied()
            .find(|candidate| accepted.iter().any(|level| level == candidate))
            .ok_or_else(|| {
                Error::admission(
                    crate::AdmissionError::Feature,
                    "prepared chat template does not accept the requested reasoning effort",
                )
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn template_default(&self, name: &str) -> Option<&Json> {
        self.defaults.get(name)
    }

    pub fn normalize_inline_system_messages(&self, messages: &mut [Json]) -> Result<()> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return Ok(());
        }
        if !messages.iter().skip(1).any(|m| m["role"] == "system") {
            return Ok(());
        }
        let supported = self.inline_system_messages.get_or_init(|| {
            let marker = "FLARE_INLINE_SYSTEM_PROBE";
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([
                    {"role":"system","content":"Instructions"},
                    {"role":"user","content":"Hello"},
                    {"role":"system","content":marker},
                    {"role":"user","content":"Continue"}
                ]),
            );
            context.insert("add_generation_prompt".into(), false.into());
            context.insert("continue_final_message".into(), false.into());
            self.template
                .as_ref()
                .unwrap()
                .render(&context, &[])
                .is_ok_and(|rendered| rendered.text.contains(marker))
        });
        if *supported {
            return Ok(());
        }
        for message in messages.iter_mut().skip(1) {
            if message["role"] != "system" {
                continue;
            }
            let invalid = || {
                Error::frontend(FrontendError::Input(Failure::new(
                    "inline system messages require text content",
                )))
            };
            let text = match &message["content"] {
                Json::String(text) => text.clone(),
                Json::Array(parts) => parts
                    .iter()
                    .map(|part| {
                        if part["type"] != "text" {
                            return Err(invalid());
                        }
                        part["text"].as_str().ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                _ => return Err(invalid()),
            };
            message["role"] = "user".into();
            message["content"] = format!("<system>{text}</system>").into();
        }
        Ok(())
    }

    pub fn load(assets: FrontendAssets, template_defaults: Map<String, Json>) -> Result<Self> {
        Self::load_encoder(assets, template_defaults, PromptEncoder::Jinja, None)
    }

    pub fn load_with_settings(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        settings: &FrontendSettings,
    ) -> Result<Self> {
        if settings.prompt_encoder != PromptEncoder::Jinja && assets.chat_template.is_some() {
            return Err(deepseek::invalid(
                "official prompt encoding does not use a chat template asset",
            ));
        }
        if settings.prompt_encoder != PromptEncoder::Jinja
            && matches!(settings.template_encoding, TemplateEncoding::TokenSegments)
        {
            return Err(deepseek::invalid(
                "official prompt encoding requires text template encoding",
            ));
        }
        Self::load_encoder(
            assets,
            template_defaults,
            settings.prompt_encoder,
            settings.image_placeholder_token_id,
        )
        .map(|frontend| frontend.with_template_encoding(settings.template_encoding))
    }

    fn load_encoder(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        prompt_encoder: PromptEncoder,
        image_placeholder_token_id: Option<u32>,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                format!("invalid tokenizer: {error}"),
                error,
            )))
        })?;
        tokenizer.with_truncation(None).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        boundaries::preserve_offsets(&mut tokenizer);
        if prompt_encoder != PromptEncoder::Jinja
            && image_placeholder_token_id.is_some_and(|id| tokenizer.id_to_token(id).is_none())
        {
            return Err(deepseek::invalid(
                "image placeholder token ID is absent from the checkpoint tokenizer",
            ));
        }
        let template_bytes = match assets.chat_template {
            Some(asset) if prompt_encoder == PromptEncoder::Jinja => asset.read()?,
            _ if prompt_encoder != PromptEncoder::Jinja => Vec::new(),
            _ => {
                return Err(deepseek::invalid(
                    "Jinja prompt encoding requires a chat template asset",
                ));
            }
        };
        let template_source = String::from_utf8(template_bytes).map_err(|error| {
            Error::frontend(FrontendError::Template(Failure::caused_by(
                format!("chat template is not UTF-8: {error}"),
                error,
            )))
        })?;
        let template = if prompt_encoder == PromptEncoder::Jinja {
            Some(templates::compile(&template_source).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
            })?)
        } else { None };
        let bos_token_id = assets
            .tokens
            .bos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token));
        let mut defaults = Map::new();
        for (name, value) in [
            ("bos_token", assets.tokens.bos_token),
            ("eos_token", assets.tokens.eos_token),
            ("unk_token", assets.tokens.unk_token),
            ("sep_token", assets.tokens.sep_token),
            ("pad_token", assets.tokens.pad_token),
            ("cls_token", assets.tokens.cls_token),
            ("mask_token", assets.tokens.mask_token),
        ] {
            if let Some(value) = value {
                defaults.insert(name.into(), Json::String(value));
            }
        }
        if let Some(tokens) = assets.tokens.additional_special_tokens {
            defaults.insert(
                "additional_special_tokens".into(),
                Json::Array(tokens.values.into_iter().map(Json::String).collect()),
            );
        }
        defaults.insert(
            "add_generation_prompt".into(),
            Json::Bool(prompt_encoder != PromptEncoder::Jinja),
        );
        defaults.insert("tools".into(), Json::Null);
        defaults.insert("documents".into(), Json::Null);
        defaults.extend(template_defaults);
        let frontend = Self {
            prompt_encoder,
            image_placeholder_token_id,
            bos_token_id,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            tokenizer,
            ordinary_tokenizer: None,
            template,
            template_source,
            defaults,
        };
        if prompt_encoder != PromptEncoder::Jinja {
            deepseek::validate_defaults(&frontend)?;
        }
        Ok(frontend)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return deepseek::render(self, messages, options);
        }
        if messages.is_empty() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Cannot apply chat template to an empty conversation. Provide at least one message.",
            ))));
        }
        self.render_jinja(messages, options)
    }

    fn render_jinja(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "continue_final_message must be a boolean or field name",
                ))));
            }
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.",
            ))));
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.",
                ))));
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Documents should be a list of dicts with 'title' and 'text' keys!",
                ))));
            }
        }
        let mut messages = messages.to_vec();
        let mut image_sources = Vec::new();
        let mut checkpoint_hints = Vec::new();
        for (index, message) in messages.iter_mut().enumerate() {
            if !message.is_object() {
                return Err(deepseek::invalid("messages must contain objects"));
            }
            if let Some(Json::Array(hints)) =
                message.as_object_mut().unwrap().remove("_checkpoint_hints")
                && self.ordinary_tokenizer.is_none()
            {
                checkpoint_hints.extend(
                    hints
                        .iter()
                        .filter_map(|hint| hint.as_u64().map(|count| (index, count as usize))),
                );
            }
            if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut) {
                for part in parts {
                    let kind = part
                        .get("type")
                        .and_then(Json::as_str)
                        .ok_or_else(|| deepseek::invalid("message content part requires type"))?;
                    if kind == "text" && !part.get("text").is_some_and(Json::is_string) {
                        return Err(deepseek::invalid("text content must be a string"));
                    }
                    if kind == "image_url" {
                        let url = part["image_url"]["url"]
                            .as_str()
                            .ok_or_else(|| deepseek::invalid("image_url.url must be a string"))?;
                        image_sources.push(url.to_owned());
                        *part = serde_json::json!({"type":"image"});
                    }
                }
            }
        }
        let final_text = continuation
            .map(|field| self.mark_continuation(&mut messages, &field))
            .transpose()?;
        context.insert("messages".into(), Json::Array(messages));
        let rendered = self
            .template
            .as_ref()
            .unwrap()
            .render(&context, &checkpoint_hints)
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.message.clone(),
                    error,
                )))
            })?;
        let marks = rendered.marks;
        let rendered = rendered.text;
        let mut segments: Vec<TokenSegment> = if self.ordinary_tokenizer.is_some() {
            serde_json::from_str(&rendered).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    "chat template must render an array of text and allow_special segments",
                    error,
                )))
            })?
        } else {
            vec![TokenSegment {
                text: rendered,
                allow_special: true,
            }]
        };
        let mut segment_ends = Vec::with_capacity(segments.len() + 1);
        segment_ends.push(0usize);
        for segment in &segments {
            segment_ends.push(segment_ends.last().unwrap() + segment.text.len());
        }
        let mut byte_boundaries = Vec::new();
        let mut hint_bytes = Vec::new();
        for mark in marks {
            match mark {
                jinja::Mark::Hint(byte) => hint_bytes.push(byte),
                jinja::Mark::Byte(byte)
                    if self.ordinary_tokenizer.is_none()
                        && byte <= *segment_ends.last().unwrap() =>
                {
                    byte_boundaries.push(byte);
                }
                jinja::Mark::Position(count) if self.ordinary_tokenizer.is_some() => {
                    byte_boundaries.push(*segment_ends.get(count).ok_or_else(|| {
                        deepseek::invalid("mark position exceeds the rendered segment count")
                    })?);
                }
                _ => {
                    return Err(deepseek::invalid(
                        "text templates require byte marks; segmented templates require segment-count marks",
                    ));
                }
            }
        }
        let mut text: String = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect();
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but the final message does not appear in the chat after applying the chat template!"))))?;
            let preserved_space = text[index + marker.len()..].starts_with(' ');
            text.truncate(index);
            if !preserved_space {
                text.truncate(text.trim_end().len());
            }
        }
        let mut remaining = text.len();
        let mut token_ids = Vec::new();
        let mut offsets = Vec::new();
        let mut byte_start = 0;
        for segment in &mut segments {
            segment.text.truncate(remaining.min(segment.text.len()));
            remaining -= segment.text.len();
            let tokenizer = if segment.allow_special {
                &self.tokenizer
            } else {
                self.ordinary_tokenizer.as_ref().unwrap_or(&self.tokenizer)
            };
            let encoding = tokenizer
                .encode(segment.text.as_str(), false)
                .map_err(|error| {
                    Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                        error.to_string(),
                        error,
                    )))
                })?;
            token_ids.extend_from_slice(encoding.get_ids());
            offsets.extend(encoding.get_offsets().iter().map(|&(start,end)| (byte_start+start,byte_start+end)));
            byte_start += segment.text.len();
            if remaining == 0 {
                break;
            }
        }
        let message_boundaries =
            boundaries::token_ends(&byte_boundaries, &hint_bytes, &offsets, text.len())?;
        Ok(RenderedPrompt {
            message_boundaries,
            text,
            token_ids,
            image_sources,
        })
    }

    fn mark_continuation(&self, messages: &mut [Json], field: &str) -> Result<String> {
        let message = messages.last_mut().ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "list index out of range",
            )))
        })?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new(format!("continue_final_message is set but the final message has no \"{field}\" to continue!")))))?;
        if !self.template_source.contains(field) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
                ),
            ))));
        }
        let text = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        } else {
            value.as_str().map(str::to_owned)
        };
        let original = text.ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but we could not find any text to continue in the final message!"))))?;
        let target = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .unwrap()
        } else {
            value
        };
        *target = Json::String(format!("{original}CONTINUE_FINAL_MESSAGE_TAG "));
        Ok(original)
    }

    #[cfg(feature = "server")]
    pub(crate) fn prepare_token_bytes(&self) -> Result<()> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map(|_| ()).map_err(Clone::clone)
    }

    pub fn token_bytes(&self, token: u32) -> Result<&[u8]> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map_err(Clone::clone)?.get(token)
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: &self.tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            finished: false,
        }
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    finished: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token_ids: &[u32]) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "incremental decoder is finished",
            ))));
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            token_ids.to_vec(),
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })
    }

    pub(crate) fn pending_text(&self) -> Result<String> {
        let text = self.tokenizer.decode(&self.ids, self.skip_special_tokens).map_err(|e| e.to_string())?;
        Ok(text.strip_prefix(&self.prefix).ok_or("decoder changed its emitted prefix")?.to_owned())
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or_else(|| {
                Error::frontend(FrontendError::Tokenizer(Failure::new(
                    "incremental decoder final text changed its emitted prefix",
                )))
            })?
            .to_owned();
        self.finished = true;
        Ok(suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_is_shareable_between_native_request_workers() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Frontend>();
    }

    fn role_frontend(guard: &str) -> Frontend {
        let source = format!(
            "{{% for message in messages %}}{guard}{{{{ message.role }}}}:{{{{ message.content }}}}\n{{% endfor %}}"
        );
        let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: None,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
        }
    }

    #[test]
    fn inline_system_roles_follow_template_capability() {
        let source = serde_json::json!([
            {"role":"system","content":"Initial instructions"},
            {"role":"user","content":"Hello"},
            {"role":"system","content":[
                {"type":"text","text":"Environment"},
                {"type":"text","text":"Use the repository"}
            ]},
            {"role":"assistant","content":"Ready"},
            {"role":"user","content":"Continue"}
        ]);
        let original = source.as_array().unwrap();
        for (guard, supported) in [
            ("", true),
            (
                "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
                false,
            ),
            (
                "{% if message.role == 'system' and not loop.first %}{% continue %}{% endif %}",
                false,
            ),
        ] {
            let frontend = role_frontend(guard);
            let mut messages = original.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(&messages[..2], &original[..2]);
            assert_eq!(&messages[3..], &original[3..]);
            if supported {
                assert_eq!(messages, *original);
            } else {
                assert_eq!(
                    messages[2],
                    serde_json::json!({
                        "role":"user","content":"<system>Environment\n\nUse the repository</system>"
                    })
                );
                frontend.render_chat(&messages, &Map::new()).unwrap();
            }
            let once = messages.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(messages, once);
        }
    }

    #[test]
    fn inline_system_conversion_preserves_text_and_rejects_nontext() {
        let frontend = role_frontend(
            "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
        );
        let mut messages = vec![
            serde_json::json!({"role":"user","content":"Hello"}),
            serde_json::json!({"role":"system","content":"\nUse <tag> & café.\n"}),
        ];
        frontend
            .normalize_inline_system_messages(&mut messages)
            .unwrap();
        assert_eq!(
            messages[1]["content"],
            "<system>\nUse <tag> & café.\n</system>"
        );
        messages[1] = serde_json::json!({"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]});
        assert!(
            frontend
                .normalize_inline_system_messages(&mut messages)
                .is_err()
        );
    }

    #[test]
    fn advisory_effort_uses_the_prepared_templates_vocabulary() {
        for (levels, expected) in [
            ("['low','medium','xhigh']", ["low", "medium", "xhigh"]),
            ("['low','high','max']", ["low", "high", "max"]),
        ] {
            let source = format!(
                "{{% if reasoning_effort not in {levels} %}}{{{{ raise_exception('unsupported effort') }}}}{{% endif %}}"
            );
            let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
            let frontend = Frontend {
                prompt_encoder: PromptEncoder::Jinja,
                image_placeholder_token_id: None,
                ordinary_tokenizer: None,
                tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                template,
                template_source: source,
                bos_token_id: None,
                vocabulary_bytes: Default::default(),
                effort_levels: Default::default(),
                inline_system_messages: Default::default(),
                defaults: Map::new(),
            };
            for (candidates, expected) in [
                (&["low"][..], expected[0]),
                (&["high", "medium"][..], expected[1]),
                (&["max", "xhigh", "high", "medium"][..], expected[2]),
            ] {
                assert_eq!(frontend.resolve_effort(candidates).unwrap(), expected);
            }
            assert!(frontend.resolve_effort(&["unsupported"]).is_err());
        }
    }

    #[test]
    fn missing_asset_retains_io_failure() {
        use std::error::Error as _;

        let asset = Asset {
            range: None,
            path: std::env::temp_dir()
                .join(format!("flare-missing-asset-{}", std::process::id()))
                .join("tokenizer.json"),
            sha256: "0".repeat(64),
        };
        let error = asset.read().unwrap_err();
        assert!(matches!(error, Error::Frontend(FrontendError::Asset(_))));
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
        let cloned = error.clone();
        assert_eq!(cloned.to_string(), error.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap(),
            cloned.source().unwrap()
        ));
    }

    #[test]
    fn render_failure_retains_template_cause_and_detail() {
        use std::error::Error as _;

        let source = "{{ raise_exception('fixture rejection') }}";
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        let frontend = Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            ordinary_tokenizer: None,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source.to_owned(),
            defaults: Map::new(),
        };
        let error = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"x"})],
                &Map::new(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, Error::Frontend(FrontendError::Template(_))));
        assert_eq!(error.to_string(), "fixture rejection");
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<jinja::Error>()
            .unwrap();
        assert_eq!(cause.message, "fixture rejection");
        let input = frontend.render_chat(&[], &Map::new()).err().unwrap();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    #[test]
    fn verified_asset_rejects_content_changed_after_fingerprinting() {
        let directory = std::env::temp_dir().join(format!("flare-asset-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tokenizer_config.json");
        let original = b"{\"eos_token\":\"<eos>\"}";
        fs::write(&path, original).unwrap();
        let asset = Asset {
            range: None,
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(original)),
        };
        assert_eq!(asset.read().unwrap(), original);
        fs::write(&path, b"{\"eos_token\":\"changed\"}").unwrap();
        assert!(matches!(
            asset.read().unwrap_err(),
            Error::Frontend(FrontendError::Asset(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
