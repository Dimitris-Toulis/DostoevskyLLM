import gradio as gr

from tokenizer import encode, decode
from LLM import *

model = LanguageModel()
m = model.to(device)

print("Loading model from file")

m.load_state_dict(torch.load("./model.pth",map_location = torch.device(device), weights_only=True))

m.eval()
print("Model weights loaded")

def generate_text(prompt, tokens_to_generate):
    encoded_prompt = encode(prompt)
    context = torch.tensor((encoded_prompt, encoded_prompt), dtype=torch.long, device=device)
    
    output = prompt[:]
    
    for encoded_token_pair in m.generate(context, max_new_tokens=tokens_to_generate, stream=True):
        text = decode([encoded_token_pair[0]])
        output += text
        yield output
    return output

demo = gr.Interface(generate_text,
                    inputs=[gr.TextArea(placeholder = "Once he"), gr.Number(value = 512, maximum = 2048, minimum = 1, step = 1, label = "Tokens to generate")],
                    outputs=[gr.TextArea(label = "Output")])

demo.launch(share = True)