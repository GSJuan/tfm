from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import string
import re
from rdkit import Chem
from rdkit import RDLogger
#RDLogger.DisableLog('rdApp.*')

'''
import os
from dotenv import load_dotenv, dotenv_values 
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
'''

class MixtralGenerationModel:
    def __init__(self, model_id, temperature=0.0, max_new_tokens=356, do_sample=False, top_k=50, top_p=0.7):
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p

        if do_sample and temperature == 0.0:
            raise ValueError(
                "`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be "
                "invalid. If you're looking for greedy decoding strategies, set `do_sample=False`")

    def __call__(self, raw_messages: str) -> str:
        """
        An example of message is:
        messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
        """
        try:
            messages = [{"role": "user", "content": raw_messages}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=len(prompt[0]) + self.max_new_tokens, use_cache=True)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generated_text
        except Exception as e:
            print(e)

            
class Nach0GenerationModel:
    def __init__(self, model_id, temperature=0.0, max_new_tokens=356, do_sample=False, top_k=50, top_p=0.7):
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        
        self.atoms_tokens = ['Ag','Al','As','Au','B','Ba','Bi','Br','C','Ca',
              'Cd','Cl','Co','Cr','Cs','Cu','F','Fe','Ga','Gd',
              'Ge','H','Hg','I','In','K','Li','M','Mg','Mn',
              'Mo','N','Na','O','P','Pt','Ru','S','Sb','Sc',
              'Se','Si','Sn','V','W','Z','Zn','c','e','n','o','p','s']
        self.atoms_tokens = sorted(self.atoms_tokens, key=lambda s: len(s), reverse=True)
        
        self.SMI_REGEX_PATTERN = r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|" + '|'.join(self.atoms_tokens) + ")"
        self.regex = re.compile(self.SMI_REGEX_PATTERN)
        
        if do_sample and temperature == 0.0:
            raise ValueError(
                "`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`")
            
    def clean_output_sequence(self, output_sequence):
        return output_sequence.replace('</s>', '').replace('<sm_', '').replace(' sm_', '').replace('>', '').strip()
    
    
    def add_special_symbols(self, text):
        output = []
        for word in text.split():
            tokens = [token for token in self.regex.findall(word)]
            if len(tokens) > 4 and (word == ''.join(tokens)) and Chem.MolFromSmiles(word):
                output.append(''.join(['<sm_'+t+'>' for t in tokens]))
            else:
                output.append(word)
        return ' '.join(output)
    
    
    def __call__(self, prompt: str) -> str:
        try:
            prompt = self.add_special_symbols(prompt)
            input_text_ids = self.tokenizer(prompt, padding="longest", truncation=True, return_tensors="pt")
            generated_text_ids = self.model.generate(**input_text_ids, do_sample=True, top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens)
            generated_text = self.tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)[0]
            print("Dirty output: " + generated_text)
            generated_text = self.clean_output_sequence(generated_text)
            return generated_text
        except Exception as e:
                print(e)


if __name__ == "__main__":
    model = GenerationModel(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.0,
        max_new_tokens=356,
        do_sample=False,
    )
    messages = "Explain what a Mixture of Experts is in less than 100 words."
    out = model(messages)
    print(out)