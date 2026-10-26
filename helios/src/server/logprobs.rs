use crate::{Result, frontend::Frontend};
use serde_json::{Value, json};

#[derive(Clone)]
pub(super) struct Formatted {
    pub entry: Value,
    pub offset: usize,
}

pub(super) fn format(
    entry: &super::token_spans::ReturnedLogprob,
    options: &super::inference::LogprobOptions,
    frontend: &Frontend,
) -> Result<Formatted> {
    let token = |id: u32, logprob: f32| -> Result<Value> {
        let bytes = frontend.token_bytes(id)?;
        Ok(json!({"token":if options.token_ids { format!("token_id:{id}") }
            else { String::from_utf8_lossy(bytes).into_owned() }, "logprob":logprob, "bytes":bytes}))
    };
    let mut value = token(entry.value.token, entry.value.logprob)?;
    let top = if options.ids.is_empty() {
        options.top
    } else {
        entry.value.top.len()
    };
    value["top_logprobs"] = Value::Array(
        entry
            .value
            .top
            .iter()
            .take(top)
            .map(|&(id, lp)| token(id, lp))
            .collect::<Result<_>>()?,
    );
    Ok(Formatted {
        entry: value,
        offset: entry.text_offset,
    })
}

pub(super) fn chat(entries: &[Formatted]) -> Value {
    json!({"content":entries.iter().map(|e| &e.entry).collect::<Vec<_>>()})
}
pub(super) fn legacy(entries: &[Formatted]) -> Value {
    let top: Vec<Value> = entries.iter().map(|e| {
        let mut map = serde_json::Map::new();
        for alternative in e.entry["top_logprobs"].as_array().unwrap() {
            map.entry(alternative["token"].as_str().unwrap().to_owned())
                .or_insert_with(|| alternative["logprob"].clone());
        }
        Value::Object(map)
    }).collect();
    json!({"tokens":entries.iter().map(|e| &e.entry["token"]).collect::<Vec<_>>(),
        "token_logprobs":entries.iter().map(|e| &e.entry["logprob"]).collect::<Vec<_>>(),
        "top_logprobs":top, "text_offset":entries.iter().map(|e| e.offset).collect::<Vec<_>>()})
}

pub(super) fn prompt(
    tokens: &[u32], values: &[Option<crate::execution::PromptLogprobs>], top: usize,
    token_ids: bool, frontend: &Frontend, offset: &mut usize, echo: bool,
) -> Result<(Vec<Value>, Value)> {
    if tokens.len() != values.len() { return Err("prompt logprob token count mismatch".into()); }
    let mut dictionaries = Vec::with_capacity(tokens.len());
    let (mut texts, mut probabilities, mut alternatives, mut offsets) = (vec![], vec![], vec![], vec![]);
    let text = |id| -> Result<String> {
        Ok(String::from_utf8_lossy(frontend.token_bytes(id)?).into_owned())
    };
    let displayed = |id, decoded: &str| if token_ids { format!("token_id:{id}") } else { decoded.to_owned() };
    let number = |lp: f32| if lp.is_finite() { lp } else { -9999. };
    for (&token, value) in tokens.iter().zip(values) {
        let decoded = if value.is_none() { text(token).unwrap_or_default() } else { text(token)? };
        let shown = displayed(token, &decoded);
        if echo {
            texts.push(json!(shown));
            offsets.push(json!(*offset));
            *offset += shown.chars().count();
        }
        let Some(value) = value else {
            if echo {
                probabilities.push(Value::Null);
                alternatives.push(Value::Null);
            }
            dictionaries.push(Value::Null);
            continue;
        };
        let mut dictionary = serde_json::Map::new();
        let mut legacy = serde_json::Map::new();
        let mut rank = 1;
        for (index, &(id, lp)) in value.value.top.iter().take(top).enumerate() {
            if index > 0 && lp < value.value.top[index - 1].1 {
                rank = index + 1;
            }
            let decoded = text(id)?;
            dictionary.insert(id.to_string(), json!({"logprob":number(lp),"rank":rank,"decoded_token":decoded}));
            if echo {
                legacy.entry(displayed(id, &decoded)).or_insert_with(|| json!(number(lp)));
            }
        }
        dictionary.insert(token.to_string(), json!({"logprob":number(value.value.logprob),"rank":value.rank,"decoded_token":decoded}));
        dictionaries.push(Value::Object(dictionary));
        if echo {
            legacy.entry(shown).or_insert_with(|| json!(number(value.value.logprob)));
            probabilities.push(json!(number(value.value.logprob)));
            alternatives.push(Value::Object(legacy));
        }
    }
    Ok((dictionaries, json!({"tokens":texts,"token_logprobs":probabilities,
        "top_logprobs":alternatives,"text_offset":offsets})))
}

pub(super) fn append_legacy(output: &mut Value, following: Value) {
    for key in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
        output[key].as_array_mut().unwrap().extend(following[key].as_array().unwrap().iter().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{execution::TokenLogprobs, output::{Event, OutputConfig, OutputPipeline, Termination, tagged::{OutputConfig as Tags, OutputStart}}};
    use crate::server::{output::Decoder, inference::LogprobOptions, protocol::{Encoder, Protocol, PromptUsage}};
    use std::sync::Arc;

    fn value(token: u32) -> TokenLogprobs {
        TokenLogprobs {
            token,
            logprob: -2.,
            top: vec![(0, -1.)],
        }
    }
    fn entries(events: &[Event]) -> Vec<super::super::token_spans::ReturnedLogprob> {
        events.iter().filter_map(|e| if let Event::TokenLogprobs(v) = e { Some(v.clone()) } else { None }).flatten().collect()
    }
    fn config(reasoning: bool) -> OutputConfig {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = reasoning.then_some(crate::model::contract::ReasoningParser {
            kind: crate::model::contract::ReasoningParserKind::ThinkTags,
            open: "<think>".into(), close: "</think>".into(), wrapping_newlines: 0, terminal_markers: vec![],
        });
        OutputConfig { syntax, control_tokens: vec![], content: Tags { structured: false,
            start: if reasoning { OutputStart::Reasoning } else { OutputStart::Content }, tools: vec![] } }
    }
    fn encoder(protocol: Protocol, frontend: Arc<Frontend>, streaming: bool) -> Encoder {
        let mut encoder = Encoder::new(protocol, "lp".into(), "model".into(), 1, 1,
            vec![PromptUsage { tokens: 1, image_tokens: 0 }], vec![], crate::server::inference::ToolChoice::Auto,
            true, false, json!({}));
        encoder.set_logprobs(
            Some(LogprobOptions {
                top: 1,
                token_ids: true,
                ids: vec![],
            }),
            frontend,
            streaming,
        );
        encoder
    }

    #[test]
    fn buffered_utf8_offsets_and_exact_bytes_match_local_and_worker_decoding() {
        let frontend = Arc::new(crate::server::stops::string_tests::frontend());
        assert_eq!(frontend.token_bytes(2).unwrap(), &[0xe4]);
        for worker in [false, true] {
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], None);
            decoder.track_logprobs(&frontend, false);
            let mut events = vec![];
            if worker {
                let mut stops = crate::server::stops::TextStops::new(&frontend, vec!["UNMATCHED".into()]).unwrap();
                stops.track(&frontend);
                for token in [2, 8] {
                    let (pieces, matched) = stops.push_logprob(token, value(token), false).unwrap();
                    assert!(!matched);
                    events.extend(decoder.push_protocol(pieces).unwrap());
                }
                let (pieces, _) = stops.finish(None).unwrap();
                events.extend(decoder.push_protocol(pieces).unwrap());
            } else {
                for token in [2, 8] {
                    events.extend(decoder.push_logprob(value(token)).unwrap());
                }
            }
            events.extend(decoder.finish(Termination::OutputLimit).unwrap());
            let entries = entries(&events);
            assert_eq!(entries.iter().map(|e| e.text_offset).collect::<Vec<_>>(), [0, 1]);
            let formatted = format(
                &entries[0],
                &LogprobOptions {
                    top: 1,
                    token_ids: false,
                    ids: vec![],
                },
                &frontend,
            )
            .unwrap();
            assert_eq!(formatted.entry["bytes"], json!([228]));
            assert_eq!(formatted.entry["token"], "�");
            let mut encoder = encoder(Protocol::Completions, frontend.clone(), false);
            encoder.apply(0, events).unwrap();
            let response = encoder.response();
            assert_eq!(response["choices"][0]["text"], "�x");
            assert_eq!(response["choices"][0]["logprobs"]["tokens"], json!(["token_id:2", "token_id:8"]));
            assert_eq!(response["choices"][0]["logprobs"]["text_offset"], json!([0,1]));
        }
    }

    #[test]
    fn stop_inside_a_multi_token_span_discards_only_tokens_starting_at_the_stop() {
        let frontend = crate::server::stops::string_tests::frontend();
        let mut stops = crate::server::stops::TextStops::new(&frontend, vec!["END".into()]).unwrap();
        stops.track(&frontend);
        assert!(stops.push_logprob(2, value(2), false).unwrap().0.is_empty());
        let (pieces, matched) = stops.push_logprob(1, value(1), false).unwrap();
        assert!(matched);
        let values: Vec<_> = pieces.iter().filter_map(|p| if let crate::output::Piece::Logprob(e) = p { Some(e.value.token) } else { None }).collect();
        assert_eq!(values, [2]);
        assert_eq!(pieces.last(), Some(&crate::output::Piece::Text("�".into())));
    }

    #[test]
    fn native_syntax_logprobs_exclude_reasoning_and_channel_markers() {
        use crate::model::contract::{ToolParser, ToolParserKind};
        for (kind, separator, boundary) in [
            (ToolParserKind::Dsml, "", "</think>"),
            (ToolParserKind::Dsml, " ", "</think>"),
            (ToolParserKind::Xtml, "", "<|close|>think<|sep|><|open|>response<|sep|>"),
        ] {
            let frontend = crate::server::output::tests::frontend(&["plan", boundary, "answer"]);
            let mut config = config(true);
            config.syntax.tools = Some(ToolParser { kind, wrapping_newlines: 0, tag_separator: separator.into() });
            config.control_tokens = vec![(1, boundary.into())];
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config));
            decoder.track_logprobs(&frontend, true);
            let mut events = Vec::new();
            for token in 0..3 {
                events.extend(decoder.push_logprob(value(token)).unwrap());
            }
            events.extend(decoder.finish(Termination::Eos).unwrap());
            let content: Vec<_> = events.iter().filter_map(|e| match e {
                Event::ContentLogprobs(entries) => Some(entries.iter().map(|e| e.value.token)),
                _ => None,
            }).flatten().collect();
            assert_eq!(content, [2], "{kind:?} {separator:?}");
        }
    }

    #[test]
    fn responses_attribute_the_tokens_first_byte_and_stream_the_same_entries() {
        let frontend = Arc::new(crate::server::output::tests::frontend(&["plan", "</", "think>OK", "fine"]));
        let mut expected = None;
        for streaming in [false, true] {
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config(true)));
            decoder.track_logprobs(&frontend, true);
            let mut encoder = encoder(Protocol::Responses, frontend.clone(), streaming);
            let mut streamed = Vec::new();
            for token in 0..4 {
                for frame in encoder.apply(0, decoder.push_logprob(value(token)).unwrap()).unwrap() {
                    if frame.data["type"] == "response.output_text.delta" {
                        streamed.extend(frame.data["logprobs"].as_array().unwrap().iter().cloned());
                    }
                }
            }
            encoder.apply(0, decoder.finish(Termination::OutputLimit).unwrap()).unwrap();
            encoder.finish(0, Termination::OutputLimit, 4, 0, None).unwrap();
            let response = encoder.response();
            let part = &response["output"][1]["content"][0];
            assert_eq!(part["text"], "OKfine");
            assert_eq!(part["logprobs"].as_array().unwrap().len(), 1);
            assert_eq!(part["logprobs"][0]["token"], "token_id:3");
            assert_eq!(json!(streamed), part["logprobs"]);
            if let Some(expected) = &expected { assert_eq!(part, expected); }
            expected = Some(part.clone());
        }
    }

    #[test]
    fn spelled_reasoning_delimiter_in_the_answer_receives_output_text_logprobs() {
        let frontend = Arc::new(crate::server::output::tests::frontend(&[
            "plan", "</think>", "Use ", "</", "think>", ".",
        ]));
        for (worker, streaming) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut config = config(true);
            config.control_tokens = vec![(1, "</think>".into())];
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config));
            decoder.track_logprobs(&frontend, true);
            let mut encoder = encoder(Protocol::Responses, frontend.clone(), streaming);
            let mut streamed = Vec::new();
            let mut stops =
                crate::server::stops::TextStops::new(&frontend, vec!["UNMATCHED".into()]).unwrap();
            stops.track(&frontend);
            for token in 0..6 {
                let events = if worker {
                    let (pieces, matched) = stops.push_logprob(token, value(token), false).unwrap();
                    assert!(!matched);
                    decoder.push_protocol(pieces).unwrap()
                } else {
                    decoder.push_logprob(value(token)).unwrap()
                };
                for frame in encoder.apply(0, events).unwrap() {
                    if frame.data["type"] == "response.output_text.delta" {
                        streamed.extend(frame.data["logprobs"].as_array().unwrap().iter().cloned());
                    }
                }
            }
            if worker {
                let (pieces, _) = stops.finish(None).unwrap();
                encoder
                    .apply(0, decoder.push_protocol(pieces).unwrap())
                    .unwrap();
            }
            for frame in encoder
                .apply(0, decoder.finish(Termination::Eos).unwrap())
                .unwrap()
            {
                if frame.data["type"] == "response.output_text.delta" {
                    streamed.extend(frame.data["logprobs"].as_array().unwrap().iter().cloned());
                }
            }
            encoder.finish(0, Termination::Eos, 6, 0, None).unwrap();
            let response = encoder.response();
            let part = &response["output"][1]["content"][0];
            assert_eq!(
                part["text"], "Use </think>.",
                "worker {worker} streaming {streaming}"
            );
            let tokens: Vec<_> = part["logprobs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["token"].clone())
                .collect();
            assert_eq!(
                tokens,
                ["token_id:2", "token_id:3", "token_id:4", "token_id:5"],
                "worker {worker} streaming {streaming}"
            );
            if streaming {
                assert_eq!(json!(streamed), part["logprobs"]);
            }
        }
    }

    #[test]
    fn streamed_token_suffix_at_finish_reuses_its_previously_delivered_entry() {
        for (reasoning, token_text) in [(true, "tail\n"), (false, "tail<")] {
            for worker in [false, true] {
                let frontend = Arc::new(crate::server::output::tests::frontend(&[token_text]));
                let mut config = config(true);
                if !reasoning {
                    config.content.start = OutputStart::Content;
                }
                let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config));
                decoder.track_logprobs(&frontend, false);
                let events = if worker {
                    let mut stops = crate::server::stops::TextStops::new(&frontend, vec!["UNMATCHED".into()]).unwrap();
                    stops.track(&frontend);
                    let (mut pieces, matched) = stops.push_logprob(0, value(0), false).unwrap();
                    assert!(!matched);
                    pieces.extend(stops.finish(None).unwrap().0);
                    decoder.push_protocol(pieces).unwrap()
                } else {
                    decoder.push_logprob(value(0)).unwrap()
                };
                let mut encoder = encoder(Protocol::Chat, frontend.clone(), true);
                let mut frames = encoder.apply(0, events).unwrap();
                let final_frames = encoder.apply(0, decoder.finish(Termination::OutputLimit).unwrap()).unwrap();
                let channel = if reasoning { "reasoning_content" } else { "content" };
                assert_eq!(final_frames.len(), 1);
                assert_eq!(final_frames[0].data["choices"][0]["delta"][channel], &token_text[4..]);
                frames.extend(final_frames);
                frames.extend(encoder.finish(0, Termination::OutputLimit, 1, 0, None).unwrap());

                let mut available = Vec::new();
                let (mut consumed, mut entries, mut pending, mut loader_missing) = (0, 0, 0, 0);
                for frame in frames {
                    let choice = &frame.data["choices"][0];
                    for entry in choice["logprobs"]["content"].as_array().into_iter().flatten() {
                        assert_eq!(entry["token"], "token_id:0");
                        available.extend(entry["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8));
                        entries += 1;
                        pending += 1;
                    }
                    if let Some(text) = choice["delta"][channel].as_str().filter(|s| !s.is_empty()) {
                        // Check source bytes cumulatively, including an earlier token's held suffix.
                        assert!(available[consumed..].starts_with(text.as_bytes()));
                        consumed += text.len();
                        loader_missing += usize::from(pending == 0);
                        pending = 0;
                    }
                }
                assert_eq!(entries, 1);
                assert_eq!(available, token_text.as_bytes());
                assert_eq!(consumed, available.len());
                assert_eq!(loader_missing, 1);
            }
        }
    }
}

#[cfg(test)]
mod stop_entry_tests {
    use crate::{execution::TokenLogprobs, output::Piece, server::stops::TextStops};
    fn value(token: u32) -> TokenLogprobs {
        TokenLogprobs {
            token,
            logprob: -2.,
            top: vec![(0, -1.)],
        }
    }
    fn tokens(pieces: &[Piece]) -> Vec<u32> {
        pieces.iter().filter_map(|p| if let Piece::Logprob(e) = p { Some(e.value.token) } else { None }).collect()
    }
    #[test]
    fn ignored_eos_survives_unmatched_stops_and_precedes_a_later_string_stop() {
        let frontend = crate::server::stops::string_tests::frontend();
        for matched in [false, true] {
            let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
            stops.track(&frontend);
            assert_eq!(tokens(&stops.push_logprob(0,value(0),false).unwrap().0), [0]);
            assert_eq!(tokens(&stops.push_logprob(10,value(10),true).unwrap().0), [10]);
            if matched {
                let (pieces, hit) = stops.push_logprob(1,value(1),false).unwrap();
                assert!(hit);
                assert!(tokens(&pieces).is_empty());
            } else {
                let (pieces, hit) = stops.push_logprob(8,value(8),false).unwrap();
                assert!(!hit);
                assert_eq!(tokens(&pieces), [8]);
            }
            assert!(tokens(&stops.finish(None).unwrap().0).is_empty());
        }
    }
    #[test]
    fn partially_clipped_token_retains_full_bytes_and_excludes_later_tokens() {
        let frontend = crate::server::output::tests::frontend(&["beforeENDafter", "later"]);
        let mut stops = TextStops::new(&frontend, vec!["END".into()]).unwrap();
        stops.track(&frontend);
        let (pieces, hit) = stops.push_logprob(0,value(0),false).unwrap();
        assert!(hit);
        assert_eq!(tokens(&pieces),[0]);
        assert_eq!(pieces.last(),Some(&Piece::Text("before".into())));
        assert_eq!(frontend.token_bytes(0).unwrap(),b"beforeENDafter");
        assert!(stops.push_logprob(1,value(1),false).unwrap().0.is_empty());
    }
}

#[cfg(test)]
mod silent_stream_tests {
    use crate::{execution::TokenLogprobs, output::{Event, OutputPipeline, Termination}, server::output::Decoder};
    #[test]
    fn ignored_eos_without_text_streams_entries_without_waiting_for_finish() {
        let frontend = crate::server::stops::string_tests::frontend();
        let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[10], None);
        decoder.track_logprobs(&frontend, false);
        for _ in 0..1024 {
            let events = decoder
                .push_logprob(TokenLogprobs {
                    token: 10,
                    logprob: -1.,
                    top: vec![],
                })
                .unwrap();
            assert!(matches!(&events[..], [Event::TokenLogprobs(entries)] if entries.len() == 1 && entries[0].value.token == 10));
        }
        assert!(decoder.finish(Termination::OutputLimit).unwrap().is_empty());
    }
}

#[cfg(test)]
mod utf8_entry_tests {
    use crate::{execution::TokenLogprobs, output::{Event, OutputPipeline, Termination}, server::output::Decoder};
    #[test]
    fn valid_replacement_character_and_long_invalid_spans_keep_each_tokens_position() {
        let frontend = crate::server::stops::string_tests::frontend();
        for tokens in [vec![11,12,13,8], [vec![2;64],vec![8]].concat()] {
            let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], None);
            decoder.track_logprobs(&frontend, true);
            let mut offsets = Vec::new();
            let mut attributed = 0;
            for token in &tokens {
                for event in decoder
                    .push_logprob(TokenLogprobs {
                        token: *token,
                        logprob: -1.,
                        top: vec![],
                    })
                    .unwrap()
                {
                    match event {
                        Event::TokenLogprobs(entries) => offsets.extend(entries.iter().map(|e| e.text_offset)),
                        Event::ContentLogprobs(entries) => attributed += entries.len(),
                        _ => {}
                    }
                }
            }
            assert!(decoder.finish(Termination::OutputLimit).unwrap().is_empty());
            assert_eq!(attributed, tokens.len());
            assert_eq!(offsets, if tokens.len() == 4 { vec![0,0,0,1] } else { (0..65).collect() });
        }
    }
}

#[cfg(test)]
mod prompt_tests {
    use super::*;
    use crate::{execution::{PromptLogprobs, TokenLogprobs}, server::{inference::ToolChoice, protocol::{Encoder, Protocol, PromptUsage}}};
    use std::sync::Arc;

    #[test]
    fn tied_prompt_tokens_have_the_same_competition_rank() {
        let frontend = crate::server::output::tests::frontend(&["A", "B", "C", "D"]);
        for token in [1, 2, 3] {
            let mut top = vec![(0, 0.); 4];
            top[..4].copy_from_slice(&[(0, -1.), (1, -2.), (2, -2.), (3, -2.)]);
            let values = [Some(PromptLogprobs { value: TokenLogprobs {
                token, logprob: -2., top,
            }, rank: 2 })];
            let (dictionary, _) = prompt(&[token], &values, 4, false, &frontend, &mut 0, true).unwrap();
            assert_eq!(dictionary[0]["0"]["rank"], 1);
            for id in ["1", "2", "3"] {
                assert_eq!(dictionary[0][id]["rank"], 2);
            }
        }
    }

    #[test]
    fn prompt_formats_nulls_actual_rank_echo_streams_and_chat_placement() {
        let frontend = Arc::new(crate::server::output::tests::frontend(&["A", " B", " C"]));
        let values = [None, Some(PromptLogprobs { value: TokenLogprobs { token: 1, logprob: -3.,
            top: vec![(2, -1.)] }, rank: 8 })];
        let (dictionary, legacy) = prompt(&[0, 1], &values, 1, true, &frontend, &mut 0, true).unwrap();
        assert_eq!(dictionary[1]["1"]["decoded_token"], " B");
        assert_eq!(dictionary[1]["2"]["decoded_token"], " C");
        assert_eq!(legacy["tokens"], json!(["token_id:0", "token_id:1"]));
        for protocol in [Protocol::Completions, Protocol::Chat] {
            let mut encoder = Encoder::new(protocol, "id".into(), "model".into(), 0, 1,
                vec![PromptUsage { tokens: 2, image_tokens: 0 }], vec![], ToolChoice::Auto, true, false, json!({}));
            encoder.set_logprobs(Some(crate::server::inference::LogprobOptions { top: 1, token_ids: false, ids: vec![] }), frontend.clone(), false);
            encoder.set_prompt_logprobs(Some(1), frontend.clone(), protocol == Protocol::Completions, vec![3]);
            encoder.prompt_logprobs(0, 0, &[0, 1], &values).unwrap();
            let response = encoder.response();
            let prompt = if protocol == Protocol::Chat { &response["prompt_logprobs"] } else { &response["choices"][0]["prompt_logprobs"] };
            assert!(prompt[0].is_null());
            assert_eq!(prompt[1]["1"]["rank"], 8);
            assert_eq!(prompt[1]["2"]["rank"], 1);
            assert_eq!(prompt[1]["1"]["decoded_token"], " B");
            if protocol == Protocol::Completions {
                let logprobs = &response["choices"][0]["logprobs"];
                assert_eq!(logprobs["tokens"], json!(["A", " B"]));
                assert_eq!(logprobs["token_logprobs"], json!([null, -3.]));
                assert_eq!(logprobs["text_offset"], json!([0, 1]));
            }
        }
        let mut encoder = Encoder::new(Protocol::Completions, "id".into(), "model".into(), 0, 1,
            vec![PromptUsage { tokens: 2, image_tokens: 0 }], vec![], ToolChoice::Auto, true, false, json!({}));
        encoder.set_logprobs(Some(crate::server::inference::LogprobOptions { top: 1, token_ids: false, ids: vec![] }), frontend.clone(), true);
        encoder.set_prompt_logprobs(Some(1), frontend, true, vec![3]);
        let frames = encoder.prompt_logprobs(0, 0, &[0, 1], &values).unwrap();
        assert_eq!(frames[0].data["choices"][0]["logprobs"]["token_logprobs"], json!([null, -3.]));
        assert!(encoder.prompt_logprobs(0, 0, &[0], &[None]).is_err());
        assert!(encoder.response()["choices"][0]["prompt_logprobs"].as_array().unwrap().is_empty());
    }
}
