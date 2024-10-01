import torch
from tokenizer import decode, encode
import LLM

model = LLM.LanguageModel()
m = model.to(LLM.device)
file = input("Model: ")

m.load_state_dict(torch.load(file,map_location = torch.device(LLM.device), weights_only=True))

m.eval()

def generate_text(prompt, tokens_to_generate):
    encoded_prompt = encode(prompt)
    context = torch.tensor((encoded_prompt, encoded_prompt), dtype=torch.long, device=LLM.device)   

    output = prompt[:]

    for encoded_token_pair in m.generate(context, max_new_tokens=tokens_to_generate, stream=True):
        text = decode([encoded_token_pair[0]])
        output += text
    return output

inp = ""
while True:
    inp = input("Input: ")
    if inp == "END": break
    tok = int(input("Tokens: "))
    print(generate_text(inp,tok))
    