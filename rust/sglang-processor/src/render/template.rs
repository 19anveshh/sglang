//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations).

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{PromptFormatter, RenderedPrompt};
use serde_json::{Value, json};

use super::{ChatFormatter, TemplateArgsRequest, TemplateError, render_oai, requested_thinking};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
}

impl TemplateArgs {
    /// The request's own `chat_template_kwargs`, then the thinking toggle its
    /// `reasoning` or `reasoning_effort` implies (`normalize_reasoning_inputs`),
    /// then the server `defaults`.
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut kwargs: HashMap<String, Value> = body["chat_template_kwargs"]
            .as_object()
            .map(|own| own.clone().into_iter().collect())
            .unwrap_or_default();
        if let Some(thinking) = requested_thinking(body) {
            for key in ["thinking", "enable_thinking"] {
                kwargs
                    .entry(key.to_owned())
                    .or_insert(Value::Bool(thinking));
            }
        }
        for (key, value) in defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        Self { kwargs }
    }
}

impl ChatFormatter {
    /// Render `request` through the formatter's own template (Jinja, Dynamo's
    /// encoder, or the legacy conversation) as SGLang's chat layer does.
    pub fn render_template(
        &self,
        request: &CreateChatCompletionRequest,
        args: &TemplateArgs,
    ) -> Result<RenderedPrompt, TemplateError> {
        let args = args.kwargs.clone();
        if let ChatFormatter::HuggingFace {
            content_parts: true,
            concatenates_text_parts,
            ..
        } = self
        {
            let request = &join_text_parts(request, *concatenates_text_parts)?;
            return self.render_prompt(&TemplateArgsRequest { request, args });
        }
        self.render_prompt(&TemplateArgsRequest { request, args })
    }
}

/// Whether the template renders a content parts list's text parts back to
/// back, as Qwen-VL's and Qwen3.5's do. The probe lists lead with an image, so
/// Dynamo hands them to the template as lists.
pub(super) fn concatenates_text_parts(formatter: &PromptFormatter) -> bool {
    let render = |texts: &[&str]| {
        let mut content =
            vec![json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})];
        content.extend(
            texts
                .iter()
                .map(|text| json!({"type": "text", "text": text})),
        );
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "",
            "messages": [{"role": "user", "content": content}],
        }))
        .ok()?;
        render_oai(formatter, &request)
            .ok()
            .map(RenderedPrompt::into_text)
    };
    let joined = render(&["One.Two."]);
    joined.is_some() && joined == render(&["One.", "Two."])
}

/// Python hands a content-parts template an all-text list as is, except a tool
/// result's, which `normalize_tool_content` joins with spaces. Dynamo joins
/// such a list with newlines unless the template needs lists, so join the tool
/// result's here, and concatenate the rest when the template would anyway.
fn join_text_parts(
    request: &CreateChatCompletionRequest,
    concatenates_text_parts: bool,
) -> Result<CreateChatCompletionRequest, TemplateError> {
    let renderer_error = |error: serde_json::Error| TemplateError::Renderer {
        message: error.to_string(),
    };
    let mut messages = serde_json::to_value(&request.messages).map_err(renderer_error)?;
    for message in messages.as_array_mut().into_iter().flatten() {
        let tool = message["role"] == "tool";
        let Some(parts) = message["content"]
            .as_array()
            .filter(|parts| parts.iter().all(|part| part["type"] == "text"))
        else {
            continue;
        };
        let texts = parts.iter().filter_map(|part| part["text"].as_str());
        if tool {
            message["content"] = texts.collect::<Vec<_>>().join(" ").into();
        } else if concatenates_text_parts && !parts.is_empty() {
            message["content"] = texts.collect::<String>().into();
        }
    }
    Ok(CreateChatCompletionRequest {
        messages: serde_json::from_value(messages).map_err(renderer_error)?,
        ..request.clone()
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Qwen2.5-VL's template iterates content parts, so Python keeps them and the
    /// template concatenates the text parts: `One.Two.`, not Dynamo's `One.\nTwo.`.
    /// A tool result's text parts join with spaces (`normalize_tool_content`).
    #[test]
    fn content_parts_template_concatenates_text_parts() {
        let template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let formatter =
            ChatFormatter::from_tokenizer_config(&json!({ "chat_template": template })).unwrap();
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "Describe."},
                ]},
                {"role": "assistant", "content": "A cat."},
                {"role": "user", "content": [
                    {"type": "text", "text": "One."},
                    {"type": "text", "text": "Two."},
                ]},
                {"role": "tool", "tool_call_id": "call_0", "content": [
                    {"type": "text", "text": "Three."},
                    {"type": "text", "text": "Four."},
                ]},
            ]
        }))
        .unwrap();
        let prompt = formatter
            .render_template(&request, &TemplateArgs::default())
            .unwrap()
            .into_text();
        assert_eq!(
            prompt,
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n\
             <|im_start|>assistant\nA cat.<|im_end|>\n\
             <|im_start|>user\nOne.Two.<|im_end|>\n\
             <|im_start|>tool\nThree. Four.<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// A template that renders each text part on its own keeps Python's lists.
    #[test]
    fn content_parts_template_keeps_separate_text_parts() {
        let template = "{% for message in messages %}{% for part in message.content %}{% if part.type == 'image' %}<image>{% else %}[{{ part.text }}]{% endif %}{% endfor %}{% endfor %}";
        let formatter =
            ChatFormatter::from_tokenizer_config(&json!({ "chat_template": template })).unwrap();
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "One."},
                {"type": "text", "text": "Two."},
            ]}]
        }))
        .unwrap();
        let prompt = formatter
            .render_template(&request, &TemplateArgs::default())
            .unwrap()
            .into_text();
        assert_eq!(prompt, "[One.][Two.]");
    }

    /// `reasoning_effort` sets both thinking toggles beneath the request's own
    /// kwargs, and the server defaults fill in only what is left.
    #[test]
    fn effort_sets_thinking_beneath_the_requests_kwargs() {
        let defaults = HashMap::from([
            ("enable_thinking".to_owned(), json!(true)),
            ("other".to_owned(), json!(1)),
        ]);
        let body = json!({
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": true},
        });
        let args = TemplateArgs::new(&body, &defaults);
        assert_eq!(
            args.kwargs,
            HashMap::from([
                ("thinking".to_owned(), json!(true)),
                ("enable_thinking".to_owned(), json!(false)),
                ("other".to_owned(), json!(1)),
            ])
        );
    }
}
