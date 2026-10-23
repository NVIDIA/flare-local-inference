use super::*;
use serde_json::json;

fn oracle(source: &str, context: &Json) -> std::result::Result<String, minijinja::Error> {
    let mut environment = minijinja::Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_filter("tojson", super::super::json::tojson as fn(_, _) -> _);
    environment.add_test("iterable", |v: &minijinja::Value| {
        !v.is_none() && v.try_iter().is_ok()
    });
    environment.add_test("sequence", |v: &minijinja::Value| {
        use minijinja::value::ValueKind;
        matches!(v.kind(), ValueKind::Seq | ValueKind::Map | ValueKind::String)
    });
    environment.set_formatter(|out, state, value| {
        if value.kind() == minijinja::value::ValueKind::Number && !value.is_integer() {
            if let Some(v) = f64::try_from(value.clone()).ok().filter(|v| v.is_finite()) {
                return out
                    .write_str(&super::super::json::python_float(v))
                    .map_err(|_| minijinja::Error::from(minijinja::ErrorKind::WriteFailure));
            }
        }
        minijinja::escape_formatter(out, state, value)
    });
    environment.add_function(
        "raise_exception",
        |v: minijinja::Value| -> std::result::Result<(), minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                v.to_string(),
            ))
        },
    );
    environment.add_template("chat", source)?;
    environment.get_template("chat")?.render(context)
}

fn compare(source: &str, context: Json) {
    let expected = oracle(source, &context).map_err(|e| e.to_string());
    let actual = CompiledTemplate::compile(source)
        .and_then(|t| t.render(context.as_object().unwrap()))
        .map(|r| r.text);
    match (expected, actual) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{source}"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("{source}\noracle: {a:?}\nowned: {b:?}"),
    }
}

#[test]
fn expressions_and_python_methods_match_minijinja() {
    let context = json!({"s":"  café 👋  ","a":[1,2,3,4],"m":{"z":true,"a":null},"d":{"b":2,"B":3,"a":1},"f":[1e-6,-0.0,1e16,2.5,3.0],"none":null});
    for expr in [
        "1+2*3",
        "(1+2)*3",
        "2**3**2",
        "not 1 == 1",
        "1 if true else 2",
        "1 if false else 2 if true else 3",
        "s|trim",
        "s.strip()",
        "s.lstrip()",
        "s.rstrip()",
        "s.split()",
        "s.split(' ', 2)",
        "s.startswith(' ')",
        "s[-4:]",
        "s[::-1]",
        "a[::-1]",
        "a[1:3]",
        "a[-100:100:2]",
        "a|length",
        "a|join(':')",
        "m.items()",
        "m|items",
        "m.get('missing', 4)",
        "m.z",
        "m['a']",
        "m|tojson",
        "[1, 2]+[3]",
        "missing|default('x')",
        "none|default('x')",
        "none|default('x',true)",
        "missing is defined",
        "missing is iterable",
        "'x' in ['x','y']",
        "'x' not in []",
        "'z' in m",
        "false and missing()",
        "true or missing()",
        "m|tojson(indent=2)",
        "m|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':'))",
        "18446744073709551615|tojson",
        "18446744073709551615",
        "9007199254740993 == 9007199254740992",
        "f[0]",
        "f[1]",
        "f[2]",
        "f[3]",
        "f[4]",
        "m is sequence",
        "s is sequence",
        "none is sequence",
        "d|dictsort",
        "d|dictsort(case_sensitive=true)",
        "d|dictsort(by='value',reverse=true)",
        "'a' \"b\" 'c'|upper",
        "['a' 'b', 'c']",
    ] {
        compare(&format!("{{{{ {expr} }}}}"), context.clone());
    }
}

#[test]
fn float_text_conversions_use_python_spelling() {
    let template = CompiledTemplate::compile("{{ f|join(' ') }}|{{ f[2]|string }}").unwrap();
    let context = json!({"f":[1e-6,-0.0,1e16,2.5,3.0]});
    assert_eq!(
        template.render(context.as_object().unwrap()).unwrap().text,
        "1e-06 -0.0 1e+16 2.5 3.0|1e+16"
    );
}

#[test]
fn scopes_macros_captures_and_control_flow_match_minijinja() {
    for source in [
        "{% set x=7 %}{% for i in range(3) %}{{ x }}{% set x=i %}{{ x }}{% endfor %}{{ x }}",
        "{% set ns=namespace(x=[]) %}{% for i in range(3) %}{% set ns.x=ns.x+[i] %}{% endfor %}{{ ns.x }}",
        "{% set ns=namespace(x=[1]) %}{% set alias=ns.x %}{% set ns.x=ns.x+[2] %}{{alias}}/{{ns.x}}",
        "{% for a,b in [('a',1),('b',2)] %}{{a}}:{{b}} {{loop.index}} {{loop.previtem}} {{loop.nextitem}}{% endfor %}",
        "{% for x in [] %}bad{% else %}empty{% endfor %}",
        "{% for x in range(6) if x % 2 == 0 %}{{x}}:{{loop.index}}/{{loop.length}}{% else %}empty{% endfor %}",
        "{% for x in range(4) %}{% if x==1 %}{% continue %}{% endif %}{% if x==3 %}{% break %}{% endif %}{{x}}{% else %}bad{% endfor %}",
        "{% macro f(x, y=3) %}{{x+y}}{% endmacro %}{{f(4)}} {{f(y=2,x=1)}}",
        "{% macro f(n) %}{% if n>0 %}{{n}}{{f(n-1)}}{% endif %}{% endmacro %}{{f(5)}}",
        "{% set x=3 %}{% macro f() %}{{x}}{% set x=4 %}{{x}}{% endmacro %}{{f()}}{{x}}",
        "{% set x=1 %}{% macro f() %}{{x}}{% endmacro %}{% set x=2 %}{{f()}}",
        "{% set x=1 %}{% macro f() %}{{x}}{% endmacro %}{% for x in [2,3] %}{{f()}}{% endfor %}",
        "{% macro f() %}{{g()}}{% endmacro %}{% macro g() %}hello{% endmacro %}{{f()}}",
        "{% set body %}hello {{3}}{% endset %}{{body|upper}}",
        "{% filter upper %}hello {{3}}{% endfilter %}",
        "a\n  {% if true %}\nb\n  {% endif %}\nc\n",
        "a\n {{- 'b' -}} \nc\n",
        "a {# ignored #} b\n",
        "a\r\nb\r\n",
    ] {
        compare(source, json!({}));
    }
}

#[test]
fn marks_are_zero_output_and_cannot_escape_captures() {
    let template = CompiledTemplate::compile(
        "é{% mark %}{% for s in ['a','👋'] %}{{s}}{% mark %}{% endfor %}{% mark 3 %}",
    )
    .unwrap();
    assert_eq!(
        template.render(&Map::new()).unwrap(),
        RenderedTemplate {
            text: "éa👋".into(),
            marks: vec![
                Mark::Byte(2),
                Mark::Byte(3),
                Mark::Byte(7),
                Mark::Position(3)
            ]
        }
    );
    assert_eq!(template.annotation_spans().len(), 3);
    assert!(std::mem::size_of::<Instruction>() <= 16);
    for source in [
        "{% macro f() %}{% mark %}{% endmacro %}",
        "{% set x %}{% mark 1 %}{% endset %}",
        "{% filter trim %}{% mark %}{% endfilter %}",
    ] {
        assert!(
            CompiledTemplate::compile(source)
                .err()
                .unwrap()
                .message
                .contains("root output")
        );
    }
    for position in ["-1", "1.5", "true", "'2'", "missing"] {
        assert!(
            CompiledTemplate::compile(&format!("{{% mark {position} %}}"))
                .unwrap()
                .render(&Map::new())
                .is_err()
        );
    }
}

#[test]
fn request_state_is_isolated_and_template_is_shareable() {
    let template=CompiledTemplate::compile("{% set ns=namespace(n=0) %}{% for x in messages %}{% set ns.n=ns.n+1 %}{{x}}{% mark %}{% endfor %}{{ns.n}}").unwrap();
    std::thread::scope(|scope| {
        for n in 0..8 {
            let template = &template;
            scope.spawn(move || {
                for _ in 0..32 {
                    let context = json!({"messages":vec!["x";n]});
                    let result = template.render(context.as_object().unwrap()).unwrap();
                    assert_eq!(result.text, format!("{}{n}", "x".repeat(n)));
                    assert_eq!(result.marks.len(), n);
                }
            });
        }
    });
}

#[test]
fn malformed_inputs_do_not_panic() {
    let alphabet = b"{}%#'\"()[]:,.=+-*/ abc012\n";
    let mut state = 17u64;
    for n in 0..10_000 {
        let source: String = (0..n % 100)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                alphabet[(state >> 32) as usize % alphabet.len()] as char
            })
            .collect();
        let _ = CompiledTemplate::compile(&source);
        let _ = CompiledTemplate::compile(&format!("{{{{{source}}}}}"));
    }
}

#[test]
fn recipe_templates_match_on_messages_tools_images_and_options() {
    let sources = [
        (
            "qwen",
            include_str!("../../../../ignition/recipes/qwen38/assets/chat_template.jinja"),
        ),
        (
            "step",
            include_str!("../../../../ignition/recipes/step5/assets/chat_template.jinja"),
        ),
        (
            "glm",
            include_str!("../../../../ignition/recipes/glm53/assets/chat_template.jinja"),
        ),
        (
            "mimo",
            include_str!("../../../../ignition/recipes/mimo_v2/assets/chat_template.jinja"),
        ),
        (
            "nemotron",
            include_str!("../../../../ignition/recipes/nemotron_h/assets/chat_template.jinja"),
        ),
        (
            "kimi",
            include_str!("../../../../ignition/recipes/kimi3/assets/chat_template.jinja"),
        ),
        (
            "gemma",
            include_str!("../../../../ignition/recipes/gemma4/assets/chat_template.google.jinja"),
        ),
        (
            "gemma-e2b-e4b",
            include_str!(
                "../../../../ignition/recipes/gemma4/assets/chat_template.google-e2b-e4b.jinja"
            ),
        ),
    ];
    let user = json!({"role":"user","content":"  café 世界 👋 a\u{301}\r\n<|im_end|>  "});
    let tools = json!([{"type":"function","function":{"name":"lookup","description":" Find <items> & café. ","parameters":{"type":"object","properties":{"query":{"type":"string","description":"A query"},"count":{"type":"integer","enum":[1,2]}},"required":["query"],"additionalProperties":false}}}]);
    let calls = json!({"role":"assistant","content":"","reasoning_content":"Need two sources.","tool_calls":[
        {"id":"b","type":"function","function":{"name":"lookup","arguments":{"query":"café 世界","count":2,"options":{"x":true,"y":null,"z":[1e-6,-0.0,1e16]}}}},
        {"id":"a","type":"function","function":{"name":"lookup","arguments":{"query":"a\nb"}}}
    ]});
    let mut histories = vec![
        json!([]),
        json!([user]),
        json!([{"role":"system","content":"Be precise."},user]),
        json!([{"role":"system","content":" "},user]),
        json!([user,{"role":"assistant","content":null},user]),
        json!([user,{"role":"assistant","content":"Answer  ","reasoning_content":" Think "},user]),
        json!([user,{"role":"assistant","content":"<think>\nThink\n</think>\nAnswer"},user]),
        json!([user,{"role":"system","content":"Inline"},user]),
        json!([user,calls,{"role":"tool","tool_call_id":"a","name":"lookup","content":"second"},{"role":"tool","tool_call_id":"b","name":"lookup","content":"first"}]),
        json!([user,calls,{"role":"tool","name":"lookup","content":[{"id":"a","output":"second"},{"id":"b","output":"first"}]}]),
    ];
    for content in [
        Json::Null,
        json!(42),
        json!({"text":"map content"}),
        json!([{"type":"text","text":"A"},{"type":"text","text":"B"}]),
        json!([{"type":"image"},{"type":"text","text":"Describe <|open|>"},{"type":"image_url","image_url":{"url":"fixture://image"}}]),
        json!([{"type":"audio"}]),
        json!(["text", "blocks"]),
    ] {
        histories.push(json!([{"role":"user","content":content}]));
    }
    let options = [
        json!({}),
        json!({"tools":tools}),
        json!({"enable_thinking":false}),
        json!({"enable_thinking":true}),
        json!({"preserve_thinking":true}),
        json!({"preserve_thinking":false,"truncate_history_thinking":true,"clear_thinking":true}),
        json!({"reasoning_effort":"low"}),
        json!({"reasoning_effort":"max"}),
        json!({"reasoning_effort":"invalid"}),
        json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"integer"}}}}}}),
        json!({"tool_choice":"required","tools":tools}),
        json!({"tool_choice":"none"}),
        json!({"add_generation_prompt":false}),
    ];
    for (name, source) in sources {
        let compiled = CompiledTemplate::compile(source).unwrap();
        let mut unmarked = source.to_owned();
        for span in compiled.annotation_spans().iter().rev() {
            unmarked.replace_range(span.clone(), "");
        }
        for (h, history) in histories.iter().enumerate() {
            for (o, option) in options.iter().enumerate() {
                let mut context = json!({"messages":history,"tools":null,"documents":null,"bos_token":"<s>","eos_token":"</s>","add_generation_prompt":true});
                context
                    .as_object_mut()
                    .unwrap()
                    .extend(option.as_object().unwrap().clone());
                let expected = oracle(&unmarked, &context);
                let actual = compiled.render(context.as_object().unwrap());
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(actual.text, expected, "{name}: history {h}, options {o}")
                    }
                    (Err(_), Err(_)) => {}
                    (expected, actual) => panic!(
                        "{name}: history {h}, options {o}\noracle: {expected:?}\nowned: {actual:?}"
                    ),
                }
            }
        }
    }
}
