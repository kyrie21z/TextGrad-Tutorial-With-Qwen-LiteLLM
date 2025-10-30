import textgrad as tg
from LiteLLMEngine import QwenEngine

# === 使用示例 ===

# 设置 API Key（请替换为你的实际密钥）
# os.environ["DASHSCOPE_API_KEY"] = "sk-...."

# 创建引擎实例
qwen_flash_engine = QwenEngine(model_name="openai/qwen-flash")
qwen_max_engine = QwenEngine(model_name="openai/qwen-max")

# 设置反向引擎（用于 textual gradients）
tg.set_backward_engine(qwen_max_engine, override=True)

# 创建模型（用于前向生成）
model = tg.BlackboxLLM(engine=qwen_flash_engine)

# 定义问题
question = tg.Variable(
    "地球为什么围绕太阳转？",
    requires_grad=False,
    role_description="a scientific question about planetary motion"
)

# 获取初始回答
answer = model(question)
answer.set_role_description("面向初中生的准确、简洁的科学解释")

print("优化前的回答：")
print(answer.value)

# 定义损失函数（评判标准）
loss_instruction = tg.Variable(
    "你是一位天文学教师。请严格评估以下回答是否科学正确、逻辑清晰、适合初中生理解。"
    "只指出错误或改进建议，不要重写答案。",
    requires_grad=False,
    role_description="loss function prompt"
)

loss_fn = tg.TextLoss(loss_instruction, qwen_max_engine)

# 优化器
optimizer = tg.TGD(parameters=[answer], engine=qwen_max_engine)

# 执行优化步骤
loss = loss_fn(answer)
print("\nLoss:", loss.value)
loss.backward()
optimizer.step()

print("\n优化后的回答：")
print(answer.value)