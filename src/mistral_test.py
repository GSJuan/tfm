from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")

print(device)

prompt = "The following string is a molecule in the SMILES format, explain it to me: CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
model = model.to(device)
with torch.no_grad():
    outputs   = model.generate(**inputs, max_length= len(prompt) + 356)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))