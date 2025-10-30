from textgrad.engine import EngineLM
from litellm import completion
import os
from typing import Optional

class QwenEngine(EngineLM):
    """
    TextGrad-compatible engine for Qwen via LiteLLM (DashScope).
    Implements the required `generate` abstract method.
    """
    def __init__(self, model_name: str = "openai/qwen-max", **kwargs):
        super().__init__()
        self.model_string = model_name
        self.kwargs = kwargs
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set DASHSCOPE_API_KEY to use Qwen via DashScope.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from Qwen using LiteLLM.
        """
        # 构造 messages：支持 system_prompt（TextGrad 会传入）
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 调用 LiteLLM
        # response = completion(
        #     model=self.model,
        #     messages=messages,
        #     **self.kwargs
        # )
        response = completion(
            model=self.model_string,
            messages=messages,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        return response["choices"][0]["message"]["content"].strip()

    def __call__(self, prompt: str, **kwargs) -> str:
        # 为了兼容性，调用 generate（无 system prompt）
        return self.generate(prompt, **kwargs)

