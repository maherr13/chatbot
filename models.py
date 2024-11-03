from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)


class Qaya(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "Qaya"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "system", "content": "You are a helpful assistant answer the Query in arabic."},
                    {"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            )

        self.dummy_response = tokenizer.decode(gen_tokens[0]).split('<|CHATBOT_TOKEN|>')[1]

        return CompletionResponse(text=self.dummy_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        messages = [{"role": "system", "content": "You are a helpful assistant answer the Query in arabic."},
                    {"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            )

        self.dummy_response = tokenizer.decode(gen_tokens[0]).split('<|CHATBOT_TOKEN|>')[1].replace('<|END_OF_TURN_TOKEN|>', "")

        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)