from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
sentence  = 'Hello World!'

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")

prompt = "Hello my name is"

inputs    = tokenizer(prompt, return_tensors="pt").to(device)
model     = model.to(device)
outputs   = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))