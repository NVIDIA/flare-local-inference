//! Text-only Decisions wire adapter; execution and normalization stay in the scorer.

use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Decisions {
    #[serde(rename = "model")]
    _model: String,
    input: Value,
    questions: Vec<Decision>,
    safety_identifier: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum Decision {
    Predicate {
        #[serde(rename = "name")]
        _name: Option<String>,
        instructions: String,
    },
    Choice {
        #[serde(rename = "name")]
        _name: Option<String>,
        instructions: String,
        choices: Vec<Choice>,
    },
    Score {
        #[serde(rename = "name")]
        _name: Option<String>,
        instructions: String,
        levels: Vec<Level>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Choice {
    value: Value,
    description: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Level {
    label: String,
    description: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputMessage {
    role: String,
    #[serde(rename = "type")]
    _kind: Option<String>,
    content: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextPart {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

pub(super) fn parse(body: Value) -> Result<(Value, Vec<Question>)> {
    if body["questions"].as_array().is_some_and(|questions| {
        questions
            .iter()
            .any(|question| question.get("name").is_some_and(|name| !name.is_string()))
    }) {
        return Err("question name must be a string when supplied".into());
    }
    let body: Decisions =
        serde_json::from_value(body).map_err(|error| Error::from(error.to_string()))?;
    if body
        .safety_identifier
        .as_ref()
        .is_some_and(|value| value.chars().count() > 128)
    {
        return Err("safety_identifier accepts at most 128 characters".into());
    }
    if !(1..=64).contains(&body.questions.len()) {
        return Err("provide 1 through 64 decision questions".into());
    }
    let input = match body.input {
        Value::String(text) => text,
        Value::Array(messages) => {
            let mut text = Vec::new();
            for value in messages {
                if value.get("type").is_some_and(|kind| kind != "message") {
                    return Err("decision input type must be message".into());
                }
                let message: InputMessage = serde_json::from_value(value)
                    .map_err(|error| Error::from(error.to_string()))?;
                if message.role != "user" {
                    return Err("decision input supports only user messages".into());
                }
                let content = match message.content {
                    Value::String(text) => text,
                    Value::Array(parts) => {
                        let mut text = Vec::new();
                        for part in parts {
                            let part: TextPart = serde_json::from_value(part)
                                .map_err(|error| Error::from(error.to_string()))?;
                            if part.kind != "input_text" {
                                return Err("decision input supports only input_text parts".into());
                            }
                            text.push(part.text);
                        }
                        text.join("\n")
                    }
                    _ => {
                        return Err(
                            "decision message content must be text or input_text parts".into()
                        );
                    }
                };
                text.push(content);
            }
            text.join("\n\n")
        }
        _ => return Err("decision input must be text or user messages".into()),
    };
    let mut questions = Vec::new();
    for question in body.questions {
        let (instructions, options) = match question {
            Decision::Predicate { instructions, .. } => (
                instructions,
                vec![
                    json!({"value":false,"description":null}),
                    json!({"value":true,"description":null}),
                ],
            ),
            Decision::Choice {
                instructions,
                choices,
                ..
            } => {
                if !(2..=128).contains(&choices.len()) {
                    return Err("choice requires 2 through 128 choices".into());
                }
                for (index, choice) in choices.iter().enumerate() {
                    if !matches!(choice.value, Value::String(_) | Value::Bool(_))
                        || choices[..index]
                            .iter()
                            .any(|other| other.value == choice.value)
                    {
                        return Err("choice values must be distinct strings or booleans".into());
                    }
                }
                let options = choices
                    .into_iter()
                    .map(|choice| json!({"value":choice.value,"description":choice.description}))
                    .collect();
                (instructions, options)
            }
            Decision::Score {
                instructions,
                levels,
                ..
            } => {
                if !(2..=10).contains(&levels.len()) {
                    return Err("score requires 2 through 10 levels".into());
                }
                (
                    instructions,
                    levels
                        .into_iter()
                        .enumerate()
                        .map(|(index, level)| {
                            let description = match level.description {
                                Some(description) => {
                                    json!({"label":level.label,"description":description})
                                }
                                None => json!(level.label),
                            };
                            json!({"value":index,"description":description})
                        })
                        .collect(),
                )
            }
        };
        questions.push(Question {
            instructions: json!(instructions),
            options,
        });
    }
    Ok((json!(input), questions))
}

pub(super) fn answer(question: &Value, probabilities: &[f64], mass: f64) -> Value {
    let winner = mode(probabilities);
    let name = &question["name"];
    match question["type"].as_str().unwrap() {
        "predicate" => json!({
            "type": "predicate",
            "name": name,
            "probability": probabilities[1],
        }),
        "choice" => {
            let choices = question["choices"].as_array().unwrap();
            let mut values = Vec::new();
            for (choice, probability) in choices.iter().zip(probabilities) {
                values.push(json!({
                    "value": choice["value"],
                    "probability": probability,
                }));
            }
            json!({
                "type": "choice",
                "name": name,
                "choice": choices[winner]["value"],
                "probabilities": values,
                "confidence": probabilities[winner] * mass,
            })
        }
        "score" => {
            let levels = question["levels"].as_array().unwrap();
            let mut values = Vec::new();
            for (index, (level, probability)) in levels.iter().zip(probabilities).enumerate() {
                values.push(json!({
                    "value": index,
                    "label": level["label"],
                    "probability": probability,
                }));
            }
            let score: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, value)| index as f64 * value)
                .sum();
            json!({
                "type": "score",
                "name": name,
                "score": score,
                "probabilities": values,
                "confidence": probabilities[winner] * mass,
            })
        }
        _ => unreachable!(),
    }
}

pub(super) fn usage(tokens: [u64; 3]) -> Value {
    json!({
        "input_tokens": tokens[0],
        "input_tokens_details": {
            "cached_tokens": tokens[1],
            "cache_write_tokens": 0,
        },
        "output_tokens": tokens[2],
        "output_tokens_details": {
            "reasoning_tokens": 0,
        },
        "total_tokens": tokens[0] + tokens[2],
    })
}
