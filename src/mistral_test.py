from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")

print(device)

prompt = """Given the following reactants and reagents, please provide a possible product. Answer only a SMILES string. 
          CCN(CC)CC.CCN=C=NCCCN(C)C.CN(C)C=O.Cl.NC1=CC=C(Cl)C=C1N.O.O=C(O)CCCCCNC(=O)C=C1C2=CC=CC=C2C2=CC=CC=C12.OC1=CC=CC2=C1N=NN2.[Cl-].[Na+]"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
model = model.to(device)
with torch.no_grad():
    outputs   = model.generate(**inputs, max_length= len(prompt) + 356)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))