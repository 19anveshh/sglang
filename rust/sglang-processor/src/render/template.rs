//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations).

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::RenderedPrompt;
use serde_json::Value;

use super::{ChatFormatter, TemplateArgsRequest, TemplateError, requested_thinking};

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
            ..
        } = self
        {
            let request = &join_text_parts(request)?;
            return self.render_prompt(&TemplateArgsRequest { request, args });
        }
        self.render_prompt(&TemplateArgsRequest { request, args })
    }
}

/// Python hands a content-parts template an all-text part list as is, and those
/// templates (Qwen-VL, Qwen3.5) concatenate the parts. Dynamo would join them
/// with newlines, so concatenate them here first.
fn join_text_parts(
    request: &CreateChatCompletionRequest,
) -> Result<CreateChatCompletionRequest, TemplateError> {
    let renderer_error = |error: serde_json::Error| TemplateError::Renderer {
        message: error.to_string(),
    };
    let mut messages = serde_json::to_value(&request.messages).map_err(renderer_error)?;
    for content in (messages.as_array_mut().into_iter().flatten())
        .filter_map(|message| message.get_mut("content"))
    {
        if let Some(parts) = content
            .as_array()
            .filter(|parts| !parts.is_empty() && parts.iter().all(|part| part["type"] == "text"))
        {
            *content = parts
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect::<String>()
                .into();
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
             <|im_start|>user\nOne.Two.<|im_end|>\n<|im_start|>assistant\n"
        );
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
