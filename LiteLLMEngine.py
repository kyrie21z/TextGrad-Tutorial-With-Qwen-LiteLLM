from textgrad.engine import EngineLM
from litellm import completion
import os
from typing import Optional, List, Union


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


class QwenVisionEngine(EngineLM):
    """
    TextGrad-compatible engine for Qwen Vision models via LiteLLM (DashScope).
    Supports multimodal inputs (text and images).
    """
    def __init__(self, model_name: str = "openai/qwen-vl-max", **kwargs):
        super().__init__()
        self.model_string = model_name
        self.kwargs = kwargs
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set DASHSCOPE_API_KEY to use Qwen Vision via DashScope.")

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from Qwen Vision model using LiteLLM.
        Supports both text and multimodal inputs.
        """
        # 构造 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 处理不同类型的内容输入
        if isinstance(content, str):
            # 纯文本输入
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            # 多模态输入（文本和图像）
            formatted_content = self._format_multimodal_content(content)
            messages.append({"role": "user", "content": formatted_content})
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # 调用 LiteLLM
        response = completion(
            model=self.model_string,
            messages=messages,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        return response["choices"][0]["message"]["content"].strip()

    def _format_multimodal_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """
        Format a list of strings and bytes into a list of dictionaries for multimodal input.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # 处理图像数据
                import base64
                # 简单判断图像类型
                if item.startswith(b'\x89\x50\x4E\x47'):
                    image_type = "png"
                elif item.startswith(b'\xFF\xD8\xFF'):
                    image_type = "jpeg"
                else:
                    # 默认当作jpeg处理
                    image_type = "jpeg"

                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                # 处理文本数据
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def __call__(self, content: Union[str, List[Union[str, bytes]]], **kwargs) -> str:
        # 为了兼容性，调用 generate
        return self.generate(content, **kwargs)