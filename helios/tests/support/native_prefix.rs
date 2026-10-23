use flare_helios::frontend::{Frontend, RenderedPrompt};
use serde_json::{Map, Value};

pub fn without_generation(
    frontend: &Frontend,
    messages: &[Value],
    options: &Map<String, Value>,
) -> RenderedPrompt {
    let mut options = options.clone();
    options.insert("add_generation_prompt".into(), true.into());
    options.insert("continue_final_message".into(), false.into());
    let mut rendered = frontend.render_chat(messages, &options).unwrap();
    let end = rendered.text.rfind("<｜Assistant｜>").unwrap();
    rendered.text.truncate(end);
    rendered.token_ids.clear();
    for (i, span) in rendered
        .text
        .split(deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN)
        .enumerate()
    {
        if i != 0 {
            rendered
                .token_ids
                .push(frontend.image_placeholder_token_id().unwrap());
        }
        rendered
            .token_ids
            .extend(frontend.encode(span, false).unwrap());
    }
    rendered
        .message_boundaries
        .retain(|&end| end < rendered.token_ids.len());
    rendered
}
