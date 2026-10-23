use std::fmt::Write;

#[cfg(test)]
use minijinja::value::{Kwargs, ViaDeserialize};
#[cfg(test)]
use minijinja::{Error, ErrorKind, Value};
use serde_json::Value as Json;

struct Formatting {
    ascii: bool,
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
}

pub(super) fn format_json(
    value: &Json,
    kwargs: &serde_json::Map<String, Json>,
) -> Result<String, String> {
    let indent = match kwargs.get("indent") {
        None | Some(Json::Null) => None,
        Some(Json::String(s)) => Some(s.clone()),
        Some(v) => Some(
            " ".repeat(
                usize::try_from(
                    v.as_i64()
                        .ok_or("indent must be an integer or string")?
                        .max(0),
                )
                .map_err(|_| "indent is too large")?,
            ),
        ),
    };
    let (item_separator, key_separator) = match kwargs.get("separators") {
        None => (
            if indent.is_some() { "," } else { ", " }.into(),
            ": ".into(),
        ),
        Some(Json::Array(v)) if v.len() == 2 => (
            v[0].as_str().ok_or("invalid separators")?.into(),
            v[1].as_str().ok_or("invalid separators")?.into(),
        ),
        _ => return Err("invalid separators".into()),
    };
    for key in kwargs.keys() {
        if !matches!(
            key.as_str(),
            "indent" | "separators" | "ensure_ascii" | "sort_keys"
        ) {
            return Err(format!("unknown keyword argument {key}"));
        }
    }
    let formatting = Formatting {
        ascii: kwargs
            .get("ensure_ascii")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        indent,
        item_separator,
        key_separator,
        sort_keys: kwargs
            .get("sort_keys")
            .and_then(Json::as_bool)
            .unwrap_or(false),
    };
    let mut output = String::new();
    formatting.write(value, 0, &mut output);
    Ok(output)
}

#[cfg(test)]
fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.to_owned())
}

#[cfg(test)]
pub(super) fn tojson(
    ViaDeserialize(value): ViaDeserialize<Json>,
    kwargs: Kwargs,
) -> Result<String, Error> {
    let indent = kwargs.get::<Option<Value>>("indent")?;
    let indent = match indent {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) if value.as_str().is_some() => Some(value.as_str().unwrap().to_owned()),
        Some(value) => {
            let width = value
                .as_i64()
                .ok_or_else(|| invalid("indent must be an integer or string"))?;
            let width =
                usize::try_from(width.max(0)).map_err(|_| invalid("indent is too large"))?;
            Some(" ".repeat(width))
        }
    };
    let separators = kwargs.get::<Option<ViaDeserialize<(String, String)>>>("separators")?;
    let (item_separator, key_separator) = match separators {
        Some(value) => value.0,
        None => (
            if indent.is_some() { "," } else { ", " }.into(),
            ": ".into(),
        ),
    };
    let formatting = Formatting {
        ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
        indent,
        item_separator,
        key_separator,
        sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
    };
    kwargs.assert_all_used()?;
    let mut output = String::new();
    formatting.write(&value, 0, &mut output);
    Ok(output)
}

impl Formatting {
    fn newline(&self, depth: usize, output: &mut String) {
        if let Some(indent) = &self.indent {
            output.push('\n');
            for _ in 0..depth {
                output.push_str(indent);
            }
        }
    }

    fn string(&self, value: &str, output: &mut String) {
        let quoted = serde_json::to_string(value).expect("strings serialize to JSON");
        if !self.ascii {
            output.push_str(&quoted);
            return;
        }
        for character in quoted.chars() {
            if character <= '\u{7e}' {
                output.push(character);
            } else {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    write!(output, "\\u{unit:04x}").expect("writing to a string succeeds");
                }
            }
        }
    }

    fn write(&self, value: &Json, depth: usize, output: &mut String) {
        match value {
            Json::Null => output.push_str("null"),
            Json::Bool(true) => output.push_str("true"),
            Json::Bool(false) => output.push_str("false"),
            Json::Number(number) if number.is_f64() => {
                output.push_str(&python_float(number.as_f64().unwrap()))
            }
            Json::Number(number) => {
                write!(output, "{number}").expect("writing to a string succeeds")
            }
            Json::String(value) => self.string(value, output),
            Json::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push_str(&self.item_separator);
                    }
                    self.newline(depth + 1, output);
                    self.write(value, depth + 1, output);
                }
                if !values.is_empty() {
                    self.newline(depth, output);
                }
                output.push(']');
            }
            Json::Object(values) => {
                output.push('{');
                let mut entries: Vec<_> = values.iter().collect();
                if self.sort_keys {
                    entries.sort_by(|left, right| left.0.cmp(right.0));
                }
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index != 0 {
                        output.push_str(&self.item_separator);
                    }
                    self.newline(depth + 1, output);
                    self.string(key, output);
                    output.push_str(&self.key_separator);
                    self.write(value, depth + 1, output);
                }
                if !values.is_empty() {
                    self.newline(depth, output);
                }
                output.push('}');
            }
        }
    }
}

pub(super) fn python_float(value: f64) -> String {
    let negative = value.is_sign_negative();
    if value == 0.0 {
        return if negative { "-0.0" } else { "0.0" }.into();
    }
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(value.abs());
    let (mantissa, exponent) = text
        .split_once('e')
        .map_or((text, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let leading = digits.len() - digits.trim_start_matches('0').len();
    let exponent = exponent + decimal - leading as i32 - 1;
    digits.drain(..leading);
    while digits.ends_with('0') {
        digits.pop();
    }
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&exponent) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!("{sign}{}{fraction}e{exponent:+03}", &digits[..1])
    } else if exponent < 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else {
        let point = (exponent + 1) as usize;
        if point >= digits.len() {
            format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::Environment;
    use serde_json::json;

    #[test]
    fn python_number_spelling_keeps_precision_and_exponent_conventions() {
        for (value, expected) in [
            (1e-6, "1e-06"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (-0.0, "-0.0"),
            (123.125, "123.125"),
        ] {
            assert_eq!(python_float(value), expected);
        }
    }

    #[test]
    fn hf_json_preserves_object_order_unicode_and_python_spacing() {
        let mut environment = Environment::new();
        environment.add_filter("tojson", tojson);
        let value = json!({"z": "<é>😀", "a": [1e-6, false, null]});
        assert_eq!(
            environment
                .render_str("{{ value|tojson }}", json!({"value": value}))
                .unwrap(),
            "{\"z\": \"<é>😀\", \"a\": [1e-06, false, null]}"
        );
        assert_eq!(
            environment
                .render_str(
                    "{{ value|tojson(ensure_ascii=true, sort_keys=true, separators=[',', ':']) }}",
                    json!({"value": value})
                )
                .unwrap(),
            "{\"a\":[1e-06,false,null],\"z\":\"<\\u00e9>\\ud83d\\ude00\"}"
        );
    }

    #[test]
    fn hf_json_indent_and_custom_separators() {
        let mut environment = Environment::new();
        environment.add_filter("tojson", tojson);
        assert_eq!(
            environment
                .render_str(
                    "{{ value|tojson(indent=2) }}",
                    json!({"value": [1, {"x": 2}]})
                )
                .unwrap(),
            "[\n  1,\n  {\n    \"x\": 2\n  }\n]"
        );
    }
}
