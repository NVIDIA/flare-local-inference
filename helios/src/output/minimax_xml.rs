//! Namespace-delimited invocations with recursively typed XML arguments.

use crate::Result;
use serde_json::{Map, Value};
use std::collections::HashSet;

const NAMESPACE: &str = "]<]minimax[>[";
const ELEMENT: &str = "]<]minimax[>[<";
const INVOKE: &str = "]<]minimax[>[<invoke name=";
const INVOKE_END: &str = "]<]minimax[>[</invoke>";

enum Element {
    Text(String),
    Children(Vec<(String, Element)>),
}

fn element(text: &str, depth: usize) -> Option<(String, Element, &str)> {
    if depth > 128 {
        return None;
    }
    let (name, mut rest) = text.strip_prefix(ELEMENT)?.split_once('>')?;
    if name.trim().is_empty() || name.starts_with('/') {
        return None;
    }
    let close = format!("{NAMESPACE}</{name}>");
    let mut children = Vec::new();
    let mut body = String::new();
    loop {
        let at = rest.find(NAMESPACE)?;
        body.push_str(&rest[..at]);
        rest = &rest[at..];
        if let Some(tail) = rest.strip_prefix(&close) {
            rest = tail;
            break;
        }
        let (child_name, child, tail) = element(rest, depth + 1)?;
        children.push((child_name, child));
        rest = tail;
    }
    let value = if children.is_empty() {
        Element::Text(body)
    } else {
        if !body.trim().is_empty() {
            let mut key = "$text".to_owned();
            while children.iter().any(|(name, _)| name == &key) {
                key.insert(0, '$');
            }
            children.push((key, Element::Text(body)));
        }
        Element::Children(children)
    };
    Some((name.to_owned(), value, rest))
}

fn container(schema: &Value) -> Option<(&str, &Value)> {
    for kind in ["array", "object"] {
        if schema["type"].as_str() == Some(kind)
            || schema["type"]
                .as_array()
                .is_some_and(|types| types.iter().any(|v| v == kind))
        {
            return Some((kind, schema));
        }
    }
    if schema.get("type").is_none() {
        if schema.get("items").is_some() {
            return Some(("array", schema));
        }
        if schema.get("properties").is_some() || schema.get("additionalProperties").is_some() {
            return Some(("object", schema));
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(choices) = schema[key].as_array() {
            for choice in choices {
                if let Some(found) = container(choice) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn convert(value: Element, schema: Option<&Value>) -> Value {
    let schema = schema.filter(|s| s.is_object());
    let shape = schema.and_then(container);
    match value {
        Element::Text(text) => {
            let mut types = Vec::new();
            super::chatml::parameter_types(schema, &mut types);
            if text.trim().is_empty() && !types.contains(&"string") {
                match shape {
                    Some(("array", _)) => return Value::Array(Vec::new()),
                    Some(("object", _)) => return Value::Object(Map::new()),
                    _ => {}
                }
            }
            super::chatml::scalar_parameter_value(&text, schema)
        }
        Element::Children(children) => match shape {
            Some(("array", schema)) => Value::Array(
                children
                    .into_iter()
                    .map(|(_, value)| convert(value, schema.get("items")))
                    .collect(),
            ),
            _ => object(children, shape.map(|(_, schema)| schema)),
        },
    }
}

fn object(elements: Vec<(String, Element)>, schema: Option<&Value>) -> Value {
    let mut fields = Map::new();
    let mut repeated = HashSet::new();
    for (name, element) in elements {
        let property = schema.and_then(|s| {
            s.get("properties")
                .and_then(|p| p.get(&name))
                .or_else(|| s.get("additionalProperties"))
        });
        let value = convert(element, property);
        if let Some(previous) = fields.get_mut(&name) {
            if repeated.insert(name.clone()) {
                *previous = Value::Array(vec![std::mem::take(previous), value]);
            } else {
                previous.as_array_mut().unwrap().push(value);
            }
        } else {
            fields.insert(name, value);
        }
    }
    Value::Object(fields)
}

pub struct MiniMaxXml;
impl super::tagged::ToolDialect for MiniMaxXml {
    const NAME: &'static str = "namespace XML";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "]<]minimax[>[<tool_call>";
    const CLOSE: &'static str = "]<]minimax[>[</tool_call>";
    const FUNCTION: Option<(&'static str, &'static str)> = Some((INVOKE, INVOKE_END));

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let mut rest = body.trim();
        let mut calls = Vec::new();
        while !rest.is_empty() {
            let invoke = rest
                .strip_prefix(INVOKE)
                .ok_or("XML tool call requires an invoke tag")?;
            let (name, tail) = invoke
                .split_once('>')
                .ok_or("unterminated XML function name")?;
            let name = name.trim().trim_matches(['\'', '"']);
            if name.is_empty() {
                return Err("tool function name is empty".into());
            }
            let schema = tools
                .iter()
                .find(|t| t["function"]["name"].as_str() == Some(name))
                .and_then(|tool| tool["function"].get("parameters"));
            let (mut parameters, tail) = tail
                .split_once(INVOKE_END)
                .ok_or("unterminated XML invocation")?;
            let mut values = Vec::new();
            while let Some((name, value, tail)) = element(parameters.trim_start(), 1) {
                values.push((name, value));
                parameters = tail;
            }
            calls.push((name.to_owned(), object(values, schema).to_string()));
            rest = tail.trim_start();
        }
        Ok(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tagged::ToolDialect;
    use serde_json::json;

    fn parse(parameters: &str, schema: Value) -> Value {
        let body = format!("{INVOKE}'lookup'>{parameters}{INVOKE_END}");
        let tools = [json!({"function":{"name":"lookup","parameters":schema}})];
        serde_json::from_str(&MiniMaxXml::parse(&body, &tools).unwrap()[0].1).unwrap()
    }

    #[test]
    fn nested_arrays_objects_empty_containers_and_repeated_fields() {
        let args = parse(
            &format!(
                "{ELEMENT}items>{ELEMENT}item>{ELEMENT}n>007{NAMESPACE}</n>{NAMESPACE}</item>{NAMESPACE}</items>\
             {ELEMENT}empty>{NAMESPACE}</empty>{ELEMENT}tag>one{NAMESPACE}</tag>\
             {ELEMENT}tag>two{NAMESPACE}</tag>{ELEMENT}tag>three{NAMESPACE}</tag>"
            ),
            json!({"properties":{
                "items":{"type":"array","items":{"type":"object","properties":{"n":{"type":"integer"}}}},
                "empty":{"anyOf":[{"type":"array"},{"type":"null"}]}
            }}),
        );
        assert_eq!(
            args,
            json!({"items":[{"n":7}],"empty":[],"tag":["one","two","three"]})
        );
    }

    #[test]
    fn mixed_text_keeps_literal_markup_and_does_not_overwrite_children() {
        let args = parse(
            &format!(
                "{ELEMENT}data>before<b>literal</b>{ELEMENT}$text>child{NAMESPACE}</$text>after{NAMESPACE}</data>"
            ),
            json!({}),
        );
        assert_eq!(
            args,
            json!({"data":{"$text":"child","$$text":"before<b>literal</b>after"}})
        );
    }

    #[test]
    fn incomplete_parameter_retains_completed_arguments_without_partial_emission() {
        let args = parse(
            &format!("{ELEMENT}a>1{NAMESPACE}</a>{ELEMENT}b>broken"),
            json!({}),
        );
        assert_eq!(args, json!({"a":"1"}));
        let nested = format!(
            "{}{}",
            format!("{ELEMENT}x>").repeat(129),
            format!("{NAMESPACE}</x>").repeat(129)
        );
        assert!(element(&nested, 1).is_none());
    }

    #[test]
    fn function_name_quotes_undeclared_and_empty_calls() {
        for name in ["lookup", "'lookup'", "\"lookup\""] {
            let body = format!("{INVOKE}{name}>{INVOKE_END}");
            assert_eq!(
                MiniMaxXml::parse(&body, &[json!({"function":{"name":"lookup"}})]).unwrap(),
                vec![("lookup".into(), "{}".into())]
            );
            assert_eq!(
                MiniMaxXml::parse(&body, &[]).unwrap(),
                vec![("lookup".into(), "{}".into())]
            );
        }
        let body = format!("{INVOKE}search>{ELEMENT}n>3{NAMESPACE}</n>{INVOKE_END}");
        let tools = [
            json!({"function":{"name":"lookup","parameters":{"properties":{"n":{"type":"integer"}}}}}),
        ];
        assert_eq!(
            MiniMaxXml::parse(&body, &tools).unwrap(),
            vec![("search".into(), "{\"n\":\"3\"}".into())]
        );
        for name in ["", "''", "\"\""] {
            let body = format!("{INVOKE}{name}>{INVOKE_END}");
            assert!(MiniMaxXml::parse(&body, &tools).is_err());
        }
    }
}
