//! One-token label scoring shared by the decision protocols.

mod adapter;

use crate::server::{GenerationEvent, HttpState, Prompt, http, inference, protocol, request};
use crate::{AdmissionError, Error, ExecutionError, Result};
use axum::{
    Extension, Json,
    extract::{FromRequest, Multipart, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{collections::HashSet, sync::Arc};
use tokio::sync::OwnedSemaphorePermit;

#[derive(Debug)]
struct Question {
    instructions: Value,
    options: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct SystemOne {
    #[serde(rename = "model")]
    _model: String,
    state: Value,
    questions: Map<String, Value>,
    #[serde(default)]
    assets: Map<String, Value>,
}

fn schema_error(location: Value, message: impl std::fmt::Display) -> (StatusCode, Value) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({"detail":[{
            "loc":location,"msg":message.to_string(),"type":"value_error"
        }]}),
    )
}

fn description(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null
    )
}

fn parse_systemone(
    body: Value,
) -> std::result::Result<(SystemOne, Vec<Question>), (StatusCode, Value)> {
    let body: SystemOne =
        serde_json::from_value(body).map_err(|error| schema_error(json!(["body"]), error))?;
    if !matches!(
        body.state,
        Value::String(_) | Value::Object(_) | Value::Array(_)
    ) {
        return Err(schema_error(
            json!(["body", "state"]),
            "state must be text, an object or an array",
        ));
    }
    if !(1..=64).contains(&body.questions.len()) {
        return Err(schema_error(
            json!(["body", "questions"]),
            "provide 1 through 64 questions",
        ));
    }
    let mut questions = Vec::new();
    for (name, value) in &body.questions {
        let invalid = |message| schema_error(json!(["body", "questions", name]), message);
        let instructions = value.get("instructions").cloned().unwrap_or(Value::Null);
        if !description(&instructions) {
            return Err(invalid(
                "instructions must be text, an object, an array or null",
            ));
        }
        let criteria = &value["criteria"];
        let options = match value["type"].as_str() {
            Some("noul") => {
                if !criteria.is_null()
                    && !criteria.as_object().is_some_and(|map| {
                        map.len() == 2 && map.contains_key("false") && map.contains_key("true")
                    })
                {
                    return Err(invalid("noul criteria must define false and true"));
                }
                ["false", "true"]
                    .into_iter()
                    .map(|name| json!({"value":name == "true","description":criteria[name]}))
                    .collect()
            }
            Some("choice") => {
                let criteria = criteria
                    .as_object()
                    .filter(|map| (1..=128).contains(&map.len()))
                    .ok_or_else(|| invalid("choice requires 1 through 128 named criteria"))?;
                criteria
                    .iter()
                    .map(|(name, value)| json!({"value":name,"description":value}))
                    .collect()
            }
            Some("score") => {
                let criteria = criteria
                    .as_array()
                    .filter(|levels| (1..=10).contains(&levels.len()))
                    .ok_or_else(|| invalid("score requires 1 through 10 ordered levels"))?;
                if criteria.iter().any(Value::is_null) {
                    return Err(invalid("score levels cannot be null"));
                }
                criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| json!({"value":index,"description":value}))
                    .collect()
            }
            _ => return Err(invalid("question type must be noul, choice or score")),
        };
        let question = Question {
            instructions,
            options,
        };
        if question
            .options
            .iter()
            .any(|option| !description(&option["description"]))
        {
            return Err(invalid(
                "criteria descriptions must be text, objects, arrays or null",
            ));
        }
        questions.push(question);
    }
    Ok((body, questions))
}

pub(in crate::server) async fn generate(
    State(state): State<Arc<HttpState>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>,
    request: Request,
) -> Response {
    let decisions = request.uri().path() == "/v1/decisions";
    if decisions
        && request
            .headers()
            .get("content-type")
            .is_some_and(|value| value.as_bytes().starts_with(b"multipart/"))
    {
        return http::error(
            protocol::Protocol::Chat,
            StatusCode::BAD_REQUEST,
            "decisions accepts text-only JSON input",
        );
    }
    let images_enabled = request
        .headers()
        .get("typesafe-danger-zone-images")
        .is_some_and(|value| value == "true");
    let permit = match state.frontend_slots.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return http::error(
                protocol::Protocol::Chat,
                StatusCode::SERVICE_UNAVAILABLE,
                "native frontend is shutting down",
            );
        }
    };
    let permit = Arc::new(permit);
    let (body, parts) = match read_body(state.max_body_bytes, request, permit.clone()).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let prepare_state = state.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let response_questions = body["questions"].clone();
        let (input, questions, assets) = if decisions {
            let (input, questions) = adapter::parse(body).map_err(|error| {
                (
                    StatusCode::BAD_REQUEST,
                    protocol::Protocol::Chat.error(StatusCode::BAD_REQUEST, error),
                )
            })?;
            (input, questions, Map::new())
        } else {
            let (parsed, questions) = parse_systemone(body)?;
            let assets = image_assets(
                &parsed.assets,
                parts,
                images_enabled,
                prepare_state.max_body_bytes,
            )
            .map_err(|error| {
                (
                    StatusCode::BAD_REQUEST,
                    protocol::Protocol::Chat.error(StatusCode::BAD_REQUEST, error),
                )
            })?;
            (parsed.state, questions, assets)
        };
        let prompts =
            prepare_questions(&prepare_state, &input, &questions, &assets).map_err(|error| {
                let error = match error {
                    Error::Admission(_) => error,
                    _ => error.admission_context(AdmissionError::Invalid),
                };
                (
                    crate::server::status(&error),
                    protocol::Protocol::Chat.failure(&error),
                )
            })?;
        Ok::<_, (StatusCode, Value)>((response_questions, prompts))
    })
    .await;
    let (response_questions, prompts) = match prepared {
        Ok(Ok(value)) => value,
        Ok(Err((status, body))) => return (status, Json(body)).into_response(),
        Err(error) => {
            return http::error(
                protocol::Protocol::Chat,
                StatusCode::INTERNAL_SERVER_ERROR,
                error,
            );
        }
    };
    let _ingress = ingress;
    match score(&state, prompts).await {
        Ok((scores, usage)) => {
            let answers = if decisions {
                Value::Array(
                    response_questions
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(scores)
                        .map(|(question, (probabilities, mass))| {
                            adapter::answer(question, &probabilities, mass)
                        })
                        .collect(),
                )
            } else {
                let answers: Map<_, _> = response_questions
                    .as_object()
                    .unwrap()
                    .iter()
                    .zip(scores)
                    .map(|((name, question), (probabilities, _))| {
                        (name.clone(), systemone_answer(question, &probabilities))
                    })
                    .collect();
                Value::Object(answers)
            };
            let usage = if decisions {
                adapter::usage(usage)
            } else {
                json!({"input_tokens":usage[0],"output_tokens":usage[2]})
            };
            Json(
                json!({"model":state.model.config().model_id,"answers":answers,
                "usage":usage}),
            )
            .into_response()
        }
        Err(error) => failure(error),
    }
}

fn failure(error: Error) -> Response {
    let error = match error {
        Error::Admission(_) | Error::Execution(_) => error,
        _ => error.admission_context(AdmissionError::Invalid),
    };
    (
        crate::server::status(&error),
        Json(protocol::Protocol::Chat.failure(&error)),
    )
        .into_response()
}

async fn read_body(
    limit: usize,
    request: Request,
    permit: Arc<OwnedSemaphorePermit>,
) -> std::result::Result<(Value, Map<String, Value>), Response> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let multipart = content_type.split(';').next() == Some("multipart/form-data");
    let encoding = request
        .headers()
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("identity")
        .to_owned();
    let (parts, body) = request.into_parts();
    let bytes = axum::body::to_bytes(body, limit).await.map_err(|error| {
        http::error(
            protocol::Protocol::Chat,
            StatusCode::PAYLOAD_TOO_LARGE,
            error,
        )
    })?;
    if !multipart {
        return tokio::task::spawn_blocking(move || {
            let _permit = permit;
            http::decode_body(&bytes, &encoding, limit)
        })
        .await
        .map_err(|error| {
            http::error(
                protocol::Protocol::Chat,
                StatusCode::INTERNAL_SERVER_ERROR,
                error,
            )
        })?
        .map(|body| (body, Map::new()))
        .map_err(|(status, message)| http::error(protocol::Protocol::Chat, status, message));
    }
    if !matches!(encoding.as_str(), "" | "identity") {
        return Err(http::error(
            protocol::Protocol::Chat,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "multipart requests require identity Content-Encoding",
        ));
    }
    let request = Request::from_parts(parts, axum::body::Body::from(bytes));
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|error| http::error(protocol::Protocol::Chat, StatusCode::BAD_REQUEST, error))?;
    let mut body = None;
    let mut images = Map::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| http::error(protocol::Protocol::Chat, StatusCode::BAD_REQUEST, error))?
    {
        let name = field.name().unwrap_or("").to_owned();
        let mime = field.content_type().unwrap_or("").to_owned();
        if name.is_empty()
            || (body.is_none() && name != "request")
            || images.contains_key(&name)
            || (body.is_some() && name == "request")
        {
            return Err(http::error(
                protocol::Protocol::Chat,
                StatusCode::BAD_REQUEST,
                "multipart requires request first and unique nonempty part names",
            ));
        }
        let bytes = field.bytes().await.map_err(|error| {
            http::error(protocol::Protocol::Chat, StatusCode::BAD_REQUEST, error)
        })?;
        if body.is_none() {
            if mime != "application/json" {
                return Err(http::error(
                    protocol::Protocol::Chat,
                    StatusCode::BAD_REQUEST,
                    "request part must have application/json content type",
                ));
            }
            let permit = permit.clone();
            body = Some(
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    http::decode_body(&bytes, "identity", limit)
                })
                .await
                .map_err(|error| {
                    http::error(
                        protocol::Protocol::Chat,
                        StatusCode::INTERNAL_SERVER_ERROR,
                        error,
                    )
                })?
                .map_err(|(status, message)| {
                    http::error(protocol::Protocol::Chat, status, message)
                })?,
            );
        } else {
            if ![
                "",
                "image/png",
                "image/jpeg",
                "image/jpg",
                "application/octet-stream",
            ]
            .contains(&mime.as_str())
            {
                return Err(http::error(
                    protocol::Protocol::Chat,
                    StatusCode::BAD_REQUEST,
                    "unsupported image part content type",
                ));
            }
            images.insert(
                name,
                Value::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
            );
        }
    }
    Ok((
        body.ok_or_else(|| {
            http::error(
                protocol::Protocol::Chat,
                StatusCode::BAD_REQUEST,
                "missing request part",
            )
        })?,
        images,
    ))
}

fn image_assets(
    assets: &Map<String, Value>,
    mut parts: Map<String, Value>,
    enabled: bool,
    limit: usize,
) -> Result<Map<String, Value>> {
    if !assets.is_empty() && !enabled {
        return Err("assets require TypeSafe-Danger-Zone-Images: true".into());
    }
    let mut result = Map::new();
    let mut reference_bytes = 0;
    for (reference, asset) in assets {
        reference_bytes += reference.len();
        if reference.trim().is_empty() || reference.len() > 1024 || reference_bytes > 65536 {
            return Err("invalid image reference length".into());
        }
        if !asset.is_object() || asset.get("type").is_some_and(|value| value != "image") {
            return Err("assets must contain image objects".into());
        }
        let data = if let Some(part) = asset.get("part") {
            let name = part.as_str().ok_or("image part must be a string")?;
            parts
                .remove(name)
                .ok_or("missing image part or part referenced more than once")?
        } else {
            if ![Some("png"), Some("jpg"), Some("jpeg")].contains(&asset["format"].as_str()) {
                return Err("base64 images require png, jpg or jpeg format".into());
            }
            asset["data_base64"].clone()
        };
        let data = data.as_str().ok_or("image data_base64 must be a string")?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| Error::from("invalid image base64"))?;
        if bytes.len() > limit {
            return Err("image exceeds the request body limit".into());
        }
        let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            "image/png"
        } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
            "image/jpeg"
        } else {
            return Err("assets support only PNG and JPEG images".into());
        };
        result.insert(
            reference.clone(),
            json!(format!("data:{mime};base64,{data}")),
        );
    }
    if !parts.is_empty() {
        return Err("unused multipart image part".into());
    }
    Ok(result)
}

fn text(parts: &mut Vec<Value>, value: &str) {
    if let Some(Value::String(previous)) = parts
        .last_mut()
        .filter(|last| last["type"] == "text")
        .and_then(|last| last.get_mut("text"))
    {
        previous.push_str(value);
    } else {
        parts.push(json!({"type":"text","text":value}));
    }
}

fn content(
    parts: &mut Vec<Value>,
    value: &Value,
    assets: &Map<String, Value>,
    used: &mut HashSet<String>,
    remaining_image_bytes: &mut usize,
) -> Result<()> {
    match value {
        Value::String(value) => {
            if let Some(url) = assets.get(value) {
                *remaining_image_bytes = remaining_image_bytes
                    .checked_sub(url.as_str().unwrap().len() + 64)
                    .ok_or("expanded decision images exceed the request body limit")?;
                parts.push(json!({"type":"image_url","image_url":{"url":url}}));
                used.insert(value.clone());
                return Ok(());
            }
            text(parts, "\"");
            let mut remaining = value.as_str();
            while let Some((position, reference, url)) = assets
                .iter()
                .filter_map(|(reference, url)| {
                    remaining
                        .find(reference)
                        .map(|position| (position, reference, url))
                })
                .min_by_key(|(position, _, _)| *position)
            {
                let escaped = serde_json::to_string(&remaining[..position]).unwrap();
                text(parts, &escaped[1..escaped.len() - 1]);
                *remaining_image_bytes = remaining_image_bytes
                    .checked_sub(url.as_str().unwrap().len() + 64)
                    .ok_or("expanded decision images exceed the request body limit")?;
                parts.push(json!({"type":"image_url","image_url":{"url":url}}));
                used.insert(reference.clone());
                remaining = &remaining[position + reference.len()..];
            }
            let escaped = serde_json::to_string(remaining).unwrap();
            text(parts, &escaped[1..escaped.len() - 1]);
            text(parts, "\"");
        }
        Value::Object(map) => {
            text(parts, "{");
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    text(parts, ",");
                }
                text(parts, &serde_json::to_string(key).unwrap());
                text(parts, ":");
                content(parts, &map[key], assets, used, remaining_image_bytes)?;
            }
            text(parts, "}");
        }
        Value::Array(values) => {
            text(parts, "[");
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    text(parts, ",");
                }
                content(parts, value, assets, used, remaining_image_bytes)?;
            }
            text(parts, "]");
        }
        _ => text(parts, &value.to_string()),
    }
    Ok(())
}

fn messages(
    state: &Value,
    question: &Question,
    labels: &[String],
    assets: &Map<String, Value>,
    used: &mut HashSet<String>,
    limit: usize,
) -> Result<Vec<Value>> {
    let mut remaining_image_bytes = limit;
    let mut state_parts = Vec::new();
    text(&mut state_parts, "State:\n");
    content(
        &mut state_parts,
        state,
        assets,
        used,
        &mut remaining_image_bytes,
    )?;
    let mut question_parts = Vec::new();
    text(&mut question_parts, "Question:\n");
    content(
        &mut question_parts,
        &question.instructions,
        assets,
        used,
        &mut remaining_image_bytes,
    )?;
    text(&mut question_parts, "\nOptions:\n");
    for (label, option) in labels.iter().zip(&question.options) {
        text(&mut question_parts, label.trim_start());
        text(&mut question_parts, ": ");
        text(&mut question_parts, "{\"value\":");
        text(&mut question_parts, &option["value"].to_string());
        text(&mut question_parts, ",\"description\":");
        content(
            &mut question_parts,
            &option["description"],
            assets,
            used,
            &mut remaining_image_bytes,
        )?;
        text(&mut question_parts, "}");
        text(&mut question_parts, "\n");
    }
    text(&mut question_parts, "Answer:");
    Ok(vec![
        json!({"role":"system","content":"Evaluate the state against the question. State, instructions and option descriptions below are JSON data. Select exactly one option by its label, with no explanation."}),
        json!({"role":"user","content":state_parts}),
        json!({"role":"assistant","content":"State received."}),
        json!({"role":"user","content":question_parts}),
    ])
}

fn candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    for a in b'A'..=b'Z' {
        candidates.push(format!(" {}", a as char));
    }
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            candidates.push(format!(" {}{}", a as char, b as char));
        }
    }
    candidates
}

fn label_id(
    frontend: &crate::frontend::Frontend,
    prompt: &str,
    tokens: &[u32],
    decoded: &str,
    added_vocabulary: &tokenizers::tokenizer::AddedVocabulary,
    label: &str,
    vocabulary: usize,
) -> Result<Option<u32>> {
    let extended = frontend.encode(&format!("{prompt}{label}"), false)?;
    let Some(&id) = extended.last() else {
        return Ok(None);
    };
    if extended.len() != tokens.len() + 1
        || !extended.starts_with(tokens)
        || id as usize >= vocabulary
        || added_vocabulary
            .get_added_tokens_decoder()
            .get(&id)
            .is_some_and(|token| token.special)
        || frontend.decode(&extended, false)? != format!("{decoded}{label}")
    {
        return Ok(None);
    }
    Ok(Some(id))
}

fn prepare_questions(
    state: &HttpState,
    input: &Value,
    questions: &[Question],
    assets: &Map<String, Value>,
) -> Result<Vec<Prompt>> {
    let config = state.model.config();
    if config.logprob_token_id_capacity == 0 {
        return Err("logprob_token_ids are not available in this model package; re-export to enable decision scoring".into());
    }
    if !assets.is_empty() && state.images.is_none() {
        return Err("image assets require a vision model package".into());
    }
    let options = json!({"enable_thinking":false,"add_generation_prompt":true})
        .as_object()
        .unwrap()
        .clone();
    let candidates = candidates();
    let added_vocabulary = state.frontend.tokenizer().get_added_vocabulary();
    let mut prompts = Vec::new();
    let mut used = HashSet::new();
    for question in questions {
        if question.options.len() > config.logprob_token_id_capacity {
            return Err("question exceeds this package's requested-ID capacity".into());
        }
        let mut labels = candidates[..question.options.len()].to_vec();
        for pass in 0..2 {
            let messages = messages(
                input,
                question,
                &labels,
                assets,
                &mut used,
                state.max_body_bytes,
            )?;
            if serde_json::to_vec(&messages).unwrap().len() > state.max_body_bytes {
                return Err("expanded decision prompt exceeds the request body limit".into());
            }
            let body = json!({"messages":messages,"max_tokens":1,"ignore_eos":true,
                "logprobs":true,"chat_template_kwargs":options});
            let mut decoded = protocol::decode(protocol::Protocol::Chat, body, false)?;
            let crate::server::inference::Input::Conversation(conversation) =
                &decoded.request.input
            else {
                unreachable!();
            };
            let mut normalized = inference::template_messages(
                conversation,
                decoded.request.tool_choice.clone(),
                &decoded.request.checkpoint_hints,
            )?;
            state
                .frontend
                .normalize_inline_system_messages(&mut normalized)?;
            let rendered = state
                .frontend
                .render_chat(&normalized, &decoded.request.options.template)?;
            let tokens = state.frontend.encode(&rendered.text, false)?;
            if tokens != rendered.token_ids {
                return Err(
                    "chat template encoding cannot verify contextual decision labels".into(),
                );
            }
            let decoded_tokens = state.frontend.decode(&tokens, false)?;
            let mut ids = Vec::new();
            let mut selected = Vec::new();
            let choices = if pass == 0 { &candidates } else { &labels };
            for label in choices {
                if let Some(id) = label_id(
                    &state.frontend,
                    &rendered.text,
                    &tokens,
                    &decoded_tokens,
                    added_vocabulary,
                    label,
                    config.contract.vocabulary,
                )? && !ids.contains(&id)
                    && !config.eos_token_ids.contains(&id)
                {
                    ids.push(id);
                    selected.push(label.clone());
                    if ids.len() == question.options.len() {
                        break;
                    }
                }
            }
            if ids.len() != question.options.len() {
                return Err(
                    "insufficient distinct single-token decision labels after the rendered prompt"
                        .into(),
                );
            }
            if pass == 0 {
                labels = selected;
                continue;
            }
            decoded.request.options.logprobs.as_mut().unwrap().ids = ids;
            let prepared = request::prepare_inference_cached(
                &state.frontend,
                state.images.as_deref(),
                &state.grammar,
                config,
                &decoded.request,
                0,
                false,
                state.model.input_store(),
            )?;
            if prepared.rendered_prompt != rendered.text
                || prepared.output.as_ref().is_none_or(|output| {
                    output.content.start != crate::output::tagged::OutputStart::Content
                })
            {
                return Err(
                    "decision scoring requires a nonthinking assistant generation prompt".into(),
                );
            }
            crate::model::admission::validate_prompt(config, &prepared.prompt)?;
            prompts.push(prepared.prompt);
        }
    }
    if used.len() != assets.len() {
        return Err("unused image asset".into());
    }
    Ok(prompts)
}

async fn score(
    state: &HttpState,
    prompts: Vec<Prompt>,
) -> Result<(Vec<(Vec<f64>, f64)>, [u64; 3])> {
    let mut answers = Vec::new();
    let mut usage = [0; 3];
    for prompt in prompts {
        let ids = prompt.logprob_token_ids.clone();
        let mut generation = state.model.generate(prompt, None)?;
        let mut scores = None;
        loop {
            match generation
                .events
                .recv()
                .await
                .ok_or_else(|| generation.failure())?
            {
                GenerationEvent::LogprobToken(value) if scores.is_none() => {
                    scores = Some(probabilities(&ids, &value)?);
                }
                GenerationEvent::Finished {
                    prompt_tokens,
                    generated_tokens,
                    reused_prompt_tokens,
                    ..
                } => {
                    if generated_tokens != 1 || scores.is_none() {
                        return Err(Error::execution(
                            ExecutionError::Completion,
                            "decision read must return exactly one scored token",
                        ));
                    }
                    usage[0] += prompt_tokens as u64;
                    usage[1] += reused_prompt_tokens;
                    usage[2] += u64::from(generated_tokens);
                    answers.push(scores.unwrap());
                    break;
                }
                GenerationEvent::AdmissionRejected(error) | GenerationEvent::Error(error) => {
                    return Err(error);
                }
                _ => {
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "unexpected decision scoring event",
                    ));
                }
            }
        }
    }
    Ok((answers, usage))
}

fn probabilities(ids: &[u32], value: &crate::execution::TokenLogprobs) -> Result<(Vec<f64>, f64)> {
    if ids.is_empty() {
        return Err(Error::execution(
            ExecutionError::Completion,
            "empty decision label selection",
        ));
    }
    let mut probabilities = Vec::new();
    for id in ids {
        let mut entries = value.top.iter().filter(|entry| entry.0 == *id);
        let logprob = entries
            .next()
            .ok_or("decision result is missing a label")?
            .1 as f64;
        if entries.next().is_some() || !logprob.is_finite() || logprob > 0.0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid decision label logprob",
            ));
        }
        probabilities.push(logprob);
    }
    let maximum = probabilities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for value in &mut probabilities {
        *value = (*value - maximum).exp();
    }
    let sum: f64 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= sum;
    }
    Ok((probabilities, maximum.exp() * sum))
}

fn mode(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..probabilities.len() {
        if probabilities[index] > probabilities[best] {
            best = index;
        }
    }
    best
}

fn systemone_answer(question: &Value, probabilities: &[f64]) -> Value {
    let n = probabilities.len();
    let winner = mode(probabilities);
    match question["type"].as_str().unwrap() {
        "noul" => json!({"type":"noul","noul":probabilities[1]}),
        "choice" => {
            let criteria = question["criteria"].as_object().unwrap();
            let distribution: Map<_, _> = criteria
                .keys()
                .zip(probabilities)
                .map(|(key, probability)| (key.clone(), json!(probability)))
                .collect();
            let confidence = if n == 1 {
                1.0
            } else {
                (n as f64 * probabilities[winner] - 1.0) / (n - 1) as f64
            };
            json!({"type":"choice","choice":criteria.keys().nth(winner).unwrap(),
                "probabilities":distribution,"confidence":confidence})
        }
        "score" => {
            let levels = question["criteria"].as_array().unwrap();
            let legend: Map<_, _> = levels
                .iter()
                .enumerate()
                .map(|(index, value)| (index.to_string(), value.clone()))
                .collect();
            let distribution: Map<_, _> = probabilities
                .iter()
                .enumerate()
                .map(|(index, value)| (index.to_string(), json!(value)))
                .collect();
            let score: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let spread: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index.abs_diff(winner) as f64 * probability)
                .sum();
            let uniform = (0..n)
                .map(|index| (index as f64 - (n - 1) as f64 / 2.0).abs())
                .sum::<f64>()
                / n as f64;
            let confidence = if n == 1 {
                1.0
            } else {
                (1.0 - spread / uniform).max(0.0)
            };
            json!({"type":"score","score":score,"legend":legend,"probabilities":distribution,"confidence":confidence})
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests;
