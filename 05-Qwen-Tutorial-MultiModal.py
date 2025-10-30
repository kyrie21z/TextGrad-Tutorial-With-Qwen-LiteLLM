import io
from PIL import Image
import textgrad as tg
from LiteLLMEngine import QwenVisionEngine

# differently from the past tutorials, we now need a multimodal LLM call instead of a standard one!
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss
import httpx

# image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
# image_data = httpx.get(image_url).content
with open("Camponotus_flavomarginatus_ant.jpg", 'rb') as image_file:
    # 读取文件的二进制内容
    image_data = image_file.read()

image_variable = tg.Variable(image_data, role_description="image to answer a question about", requires_grad=False)

question_variable = tg.Variable("What do you see in this image?", role_description="question", requires_grad=False)

qwen_llm_engine = QwenVisionEngine(model_name="openai/qwen-vl-plus")
tg.set_backward_engine(qwen_llm_engine)

response = MultimodalLLMCall(engine=qwen_llm_engine)([image_variable, question_variable])
print("# before textgrad:\n")
print(response.value)

Image.open(io.BytesIO(image_data))

loss_fn = ImageQALoss(
    evaluation_instruction="Does this seem like a complete and good answer for the image? Criticize. Do not provide a new answer.",
    engine=qwen_llm_engine
)
loss = loss_fn(question=question_variable, image=image_variable, response=response)
print("\n\n# loss:\n")
print(loss.value)

optimizer = tg.TGD(parameters=[response])
loss.backward()
optimizer.step()
print("\n\n# after textgrad:\n")
print(response.value)