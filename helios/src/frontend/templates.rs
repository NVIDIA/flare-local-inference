use super::jinja::{CompiledTemplate, Error};

const QWEN: &str = include_str!("../../../ignition/recipes/qwen38/assets/chat_template.jinja");
const STEP: &str = include_str!("../../../ignition/recipes/step5/assets/chat_template.jinja");

pub(super) fn compile(source: &str) -> Result<CompiledTemplate, Error> {
    for annotated in [QWEN, STEP] {
        if source == annotated.replace("{%- mark -%}", "") {
            return CompiledTemplate::compile(annotated);
        }
    }
    let compiled = CompiledTemplate::compile(source)?;
    if compiled.annotation_spans().is_empty() {
        return Err(Error { message: "chat template requires message-boundary mark annotations; only the deployed Qwen3.8 and Step templates can be upgraded automatically".into(), offset: 0 });
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_requires_exact_legacy_bytes() {
        for annotated in [QWEN, STEP] {
            let raw = annotated.replace("{%- mark -%}", "");
            assert!(!compile(&raw).unwrap().annotation_spans().is_empty());
            assert!(compile(&(raw + " ")).is_err());
            assert!(compile(annotated).is_ok());
        }
        assert!(compile("{{ messages }}").is_err());
    }
}
