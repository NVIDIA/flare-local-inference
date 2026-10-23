//! Brace-delimited calls with bare keys and `<|"|>`-quoted strings.

use crate::Result;
use serde_json::{Map, Value};

const QUOTE: &str = "<|\"|>";

fn quoted<'a>(text: &mut &'a str) -> Option<Result<&'a str>> {
    let current: &'a str = text;
    let body = current.strip_prefix(QUOTE)?;
    Some(match body.split_once(QUOTE) {
        Some((value, rest)) => {
            *text = rest;
            Ok(value)
        }
        None => Err("unterminated tool call string".into()),
    })
}

fn value(text: &mut &str, schema: Option<&Value>, depth: usize) -> Result<Value> {
    if depth > 128 {
        return Err("tool call arguments exceed the nesting bound".into());
    }
    *text = text.trim_start();
    if let Some(string) = quoted(text) {
        return Ok(Value::String(string?.into()));
    }
    let current = *text;
    let close = if let Some(rest) = current.strip_prefix('{') {
        *text = rest;
        '}'
    } else if let Some(rest) = current.strip_prefix('[') {
        *text = rest;
        ']'
    } else {
        let end = current.find([',', '}', ']']).unwrap_or(current.len());
        let raw = current[..end].trim();
        *text = &current[end..];
        if raw.is_empty() {
            return Err("tool call argument is empty".into());
        }
        let string = schema.is_some_and(|s| s["type"] == "string");
        return Ok(match serde_json::from_str::<Value>(raw) {
            Ok(value) if !string && !value.is_object() && !value.is_array() => value,
            _ => Value::String(raw.into()),
        });
    };
    let mut fields = Map::new();
    let mut items = Vec::new();
    loop {
        *text = text.trim_start();
        if let Some(rest) = text.strip_prefix(close) {
            *text = rest;
            break;
        }
        if !(fields.is_empty() && items.is_empty()) {
            *text = text
                .strip_prefix(',')
                .ok_or("tool call arguments require comma separators")?
                .trim_start();
        }
        if close == ']' {
            items.push(value(text, schema.and_then(|s| s.get("items")), depth + 1)?);
            continue;
        }
        let key = match quoted(text) {
            Some(key) => key?.to_owned(),
            None => {
                let current = *text;
                let (key, _) = current
                    .split_once(':')
                    .ok_or("tool call argument requires a key")?;
                if key.contains([',', '{', '}', '[', ']']) {
                    return Err("tool call argument requires a key".into());
                }
                *text = &current[key.len()..];
                key.trim().to_owned()
            }
        };
        *text = text
            .trim_start()
            .strip_prefix(':')
            .ok_or("tool call argument requires a key separator")?;
        if key.is_empty() {
            return Err("tool call argument key is empty".into());
        }
        let property = schema.and_then(|s| {
            s.get("properties")
                .and_then(|p| p.get(&key))
                .or_else(|| s.get("additionalProperties"))
        });
        let parsed = value(text, property, depth + 1)?;
        fields.insert(key, parsed);
    }
    Ok(if close == ']' {
        Value::Array(items)
    } else {
        Value::Object(fields)
    })
}

pub struct GemmaCall;
impl super::tagged::ToolDialect for GemmaCall {
    const NAME: &'static str = "Gemma call";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<|tool_call>";
    const CLOSE: &'static str = "<tool_call|>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let body = body.trim_start();
        let call = body
            .strip_prefix("call:")
            .or_else(|| body.strip_prefix(':'))
            .ok_or("tool call requires a call: prefix")?;
        let (name, _) = call
            .split_once('{')
            .ok_or("tool call requires brace-delimited arguments")?;
        let mut rest = &call[name.len()..];
        let name = name.trim();
        if name.is_empty() {
            return Err("tool function name is empty".into());
        }
        let schema = tools
            .iter()
            .filter_map(|tool| tool.get("function"))
            .find(|function| function.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|function| function.get("parameters"));
        let arguments = value(&mut rest, schema, 0)?;
        if !rest.trim().is_empty() {
            return Err("unexpected content after tool call arguments".into());
        }
        Ok(vec![(name.to_owned(), arguments.to_string())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tagged::ToolDialect;
    use serde_json::json;

    fn parse(body: &str, tools: &[Value]) -> Result<(String, Value)> {
        let (name, arguments) = GemmaCall::parse(body, tools)?.remove(0);
        Ok((name, serde_json::from_str(&arguments).unwrap()))
    }

    #[test]
    fn nested_values_follow_the_template_argument_syntax() {
        let body = "call:plan{a:<|\"|>x, {y}: [z]<|\"|>,b:3,c:2.50,d:true,e:null,\
                    f:[1,<|\"|>two<|\"|>,{g:[]}],h:{<|\"|>quoted key<|\"|>:-1e3},i:{}}";
        assert_eq!(
            parse(body, &[]).unwrap(),
            (
                "plan".into(),
                json!({"a":"x, {y}: [z]","b":3,"c":2.5,"d":true,"e":null,
                       "f":[1,"two",{"g":[]}],"h":{"quoted key":-1e3},"i":{}})
            )
        );
    }

    #[test]
    fn declared_string_schemas_keep_bare_literals_as_text() {
        let tools = [json!({"function":{"name":"lookup","parameters":{"properties":{
            "zip":{"type":"string"},"n":{"type":"integer"},
            "tags":{"type":"array","items":{"type":"string"}}}}}})];
        let body = "call:lookup{zip:02139,n:7,tags:[1,true],word:bare text}";
        assert_eq!(
            parse(body, &tools).unwrap().1,
            json!({"zip":"02139","n":7,"tags":["1","true"],"word":"bare text"})
        );
        let (name, args) = parse("call:missing{zip:02139,n:7}", &tools).unwrap();
        assert_eq!((name.as_str(), args), ("missing", json!({"zip":"02139","n":7})));
    }

    #[test]
    fn bare_openers_and_empty_arguments() {
        assert_eq!(parse("\n:lookup{}", &[]).unwrap(), ("lookup".into(), json!({})));
        assert_eq!(parse("call:lookup{ }", &[]).unwrap(), ("lookup".into(), json!({})));
    }

    #[test]
    fn malformed_calls_are_rejected() {
        for body in [
            "lookup{}",
            "call:{}",
            "call:lookup",
            "call:lookup{a:1",
            "call:lookup{a:<|\"|>open}",
            "call:lookup{a:<|\"|>x<|\"|> b:2}",
            "call:lookup{a:}",
            "call:lookup{:1}",
            "call:lookup{a}",
            "call:lookup{a,b:1}",
            "call:lookup{}trailing",
        ] {
            assert!(GemmaCall::parse(body, &[]).is_err(), "{body}");
        }
        let nested = format!("call:lookup{{a:{}{}}}", "[".repeat(130), "]".repeat(130));
        assert!(GemmaCall::parse(&nested, &[]).is_err());
    }
}
