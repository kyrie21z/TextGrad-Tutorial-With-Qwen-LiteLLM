from litellm import completion
import os

# os.environ["DASHSCOPE_API_KEY"] = "sk-edf0d38f077043a58050bf09e1bf9739"

response = completion(
    model="openai/qwen-max",
    messages=[{"role": "user", "content": "hello from litellm"}],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
print(response)
