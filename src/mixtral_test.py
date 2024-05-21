from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

'''
import os
from dotenv import load_dotenv, dotenv_values 
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
'''


class GenerationModel:
    def __init__(self, model_id, temperature=0.0, max_new_tokens=356, do_sample=False, top_k=50, top_p=0.7):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bits=True)
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