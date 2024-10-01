import torch
from tokenizer import encode,decode
import LLM
import os
import re
import json

block_size = LLM.block_size
device = LLM.device

# Train parameters
batch_size = 56
eval_interval = 500
eval_iters = 200
max_iters = 20000
learning_rate = 3e-4

torch.cuda.empty_cache()

train_data = torch.tensor([],dtype=torch.long)
val_data = torch.tensor([],dtype=torch.long)

def prepare_data(text):
    return re.sub("\\n+","\\n\\n",re.sub("(?<!\\n)\n(?!\\n)"," ",text))

print("Preparing data...")
with os.scandir("./data/") as it:
    for entry in it:
        if entry.name.endswith(".txt") and entry.is_file():
            data = encode(prepare_data(open(entry.path, "r").read()))
            n = int(0.9*len(data))
            train_data = torch.cat((train_data, torch.tensor(data[:n], dtype=torch.long)), 0)
            val_data = torch.cat((val_data, torch.tensor(data[n:], dtype=torch.long)), 0)

print("Data prepared")

with open("parameters.json","w") as params_file:
    params_file.write(json.dumps({
        "n_embd": LLM.n_embd,
        "n_head": LLM.n_head,
        "n_layer": LLM.n_layer,
        "block_size": LLM.block_size,
        "dropout": LLM.dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
    }, indent=4))

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = LLM.LanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter % 100 == 0:
        print(f"step {iter}")

    if iter % 1000 == 0 and iter != 0:
        torch.save(model.state_dict(), f"model_{iter}.pth")


    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pth")