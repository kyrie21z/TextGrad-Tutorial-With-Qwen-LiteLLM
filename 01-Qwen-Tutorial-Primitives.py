from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.loss import TextLoss
# from dotenv import load_dotenv
# load_dotenv()

from LiteLLMEngine import QwenEngine

x = Variable("A sntence with a typo", role_description="The input sentence", requires_grad=True)
print("# value of x before textgrad:\n")
print(x.value)

engine = QwenEngine()

print("\n\n# greeting to Qwen:\n")
print(engine.generate("Hello how are you?"))

system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
loss = TextLoss(system_prompt, engine=engine)

optimizer = TextualGradientDescent(parameters=[x], engine=engine)

l = loss(x)
print("\n\n# loss: \n", l.value)
l.backward(engine)
optimizer.step()

print("\n\n# value of x after textgrad:\n")
print(x.value)

optimizer.zero_grad()