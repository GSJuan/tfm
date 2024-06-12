from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import re
from rdkit import Chem

class BaseGenerationModel:
    
    def __init__(self, model_id, tokenizer_id=None, temperature=0.0, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, device_map="auto")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if do_sample and temperature == 0.0:
            raise ValueError(
                "`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be "
                "invalid. If you're looking for greedy decoding strategies, set `do_sample=False`")

    def generate_text(self, inputs):
        raise NotImplementedError("Subclasses should implement this method")

            
class MistralGenerationModel(BaseGenerationModel):
   
    def __init__(self, model_id, temperature=1, max_new_tokens=1000, do_sample=True, top_k=50, top_p=0.7):
        super().__init__(model_id, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        
    def generate_text(self, prompt):
        try:
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs   = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, do_sample=self.do_sample)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True, use_cache=True)
        except Exception as e:
            print(e)
            
            
class MixtralGenerationModel(BaseGenerationModel):
    def __init__(self, model_id, temperature=1, max_new_tokens=1000, do_sample=True, top_k=50, top_p=0.7):
        super().__init__(model_id, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

    def generate_text(self, prompt):
        try:
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, use_cache=True)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generated_text
        except Exception as e:
            print(e)

class Nach0GenerationModel(BaseGenerationModel):
    def __init__(self, model_id, temperature=1, max_new_tokens=512, do_sample=True, top_k=100, top_p=0.95):
        super().__init__(model_id, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
        self.atoms_tokens = sorted(['Ag','Al','As','Au','B','Ba','Bi','Br','C','Ca',
                                    'Cd','Cl','Co','Cr','Cs','Cu','F','Fe','Ga','Gd',
                                    'Ge','H','Hg','I','In','K','Li','M','Mg','Mn',
                                    'Mo','N','Na','O','P','Pt','Ru','S','Sb','Sc',
                                    'Se','Si','Sn','V','W','Z','Zn','c','e','n','o','p','s'],
                                   key=lambda s: len(s), reverse=True)
        self.SMI_REGEX_PATTERN = r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|" + '|'.join(self.atoms_tokens) + ")"
        self.regex = re.compile(self.SMI_REGEX_PATTERN)

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

    def generate_text(self, inputs):
        try:
            prompt = self.add_special_symbols(inputs)
            input_text_ids = self.tokenizer(prompt, padding="longest", truncation=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated_text_ids = self.model.generate(**input_text_ids, do_sample=self.do_sample, top_k=self.top_k, top_p=self.top_p, max_length=self.max_new_tokens)
                generated_text = self.tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)[0]
                #print("Dirty output: "+ generated_text)
                generated_text = self.clean_output_sequence(generated_text)
                #print(generated_text)
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
    out = model.generate_text(messages)
    print(out)