use super::*;
use crate::engine::Tokenized;

impl VllmEngine {
    pub(super) async fn prefill_chat(&self, messages: &[ChatMessage]) -> Result<(), EngineError> {
        if messages.is_empty() {
            return Ok(());
        }
        let tokenized = self.tokenize_chat(messages).await?;
        if tokenized.ids.len() >= tokenized.limit as usize {
            return Err(EngineError::ContextWindowExceeded {
                limit: tokenized.limit,
            });
        }
        // Installed vLLM supports echo + max_tokens=0 for prefill without accepting
        // generated output. The echo is consumed privately and never logged or streamed.
        let response = self.client.post(format!("{}/v1/completions",self.base_url))
            .json(&serde_json::json!({"model":self.model,"prompt":tokenized.ids,"echo":true,"max_tokens":0,"stream":false}))
            .send().await?.error_for_status()?;
        let response: serde_json::Value = response.json().await?;
        if !response["choices"].is_array() {
            return Err(EngineError::BadResponse(
                "portable prefill returned an invalid response".into(),
            ));
        }
        Ok(())
    }

    async fn token_request(&self, body: serde_json::Value) -> Result<Tokenized, EngineError> {
        let response = self
            .client
            .post(format!("{}/tokenize", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        #[derive(serde::Deserialize)]
        struct Tokens {
            tokens: Vec<u32>,
            count: usize,
            max_model_len: u32,
        }
        let tokens: Tokens = response.json().await?;
        if tokens.count != tokens.tokens.len() || tokens.max_model_len == 0 {
            return Err(EngineError::BadResponse(
                "inconsistent tokenizer response".into(),
            ));
        }
        Ok(Tokenized {
            ids: tokens.tokens,
            limit: self.profile.map_or(tokens.max_model_len, |p| {
                p.max_context.min(tokens.max_model_len)
            }),
        })
    }
    pub(super) async fn tokenize_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Tokenized, EngineError> {
        let messages: Vec<_> = messages
            .iter()
            .map(|m| serde_json::json!({"role":m.role.as_str(),"content":m.content}))
            .collect();
        self.token_request(serde_json::json!({"model":self.model,"messages":messages,"add_generation_prompt":true})).await
    }
    pub(super) async fn encode_text(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        Ok(self
            .token_request(
                serde_json::json!({"model":self.model,"prompt":text,"add_special_tokens":false}),
            )
            .await?
            .ids)
    }
}
