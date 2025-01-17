import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf
import time
from metrics import is_valid_smiles

from utils import get_project_root

class BaseCFGModel:
    
    def __init__(self, model_id, tokenizer_id=None, temperature=0.0, max_new_tokens=50, do_sample=False, top_k=50, top_p=0.7, repetition_penalty=1.3, num_return_sequences=1, load_in_4bit=False):

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty=repetition_penalty
        self.num_return_sequences=num_return_sequences
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, device_map="auto")
        self.load_in_4bit = load_in_4bit
        
        if do_sample and temperature == 0.0:
            raise ValueError(
                "`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be "
                "invalid. If you're looking for greedy decoding strategies, set `do_sample=False`")

    def generate_text(self, inputs):
        raise NotImplementedError("Subclasses should implement this method")

        
class MistralCFGModel(BaseCFGModel):
   
    def __init__(self, model_id, temperature=1, max_new_tokens=30, do_sample=False, top_k=50, top_p=0.7, repetition_penalty=1.3, num_return_sequences=1, load_in_4bit=False):
        super().__init__(model_id, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, load_in_4bit=load_in_4bit)
        
        self.model_type = "base"
        if model_id.find("Instruct") != -1:
            self.model_type = "instruct"  
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=self.load_in_4bit)
        
        # Load grammar
        grammar_name = "basic"
        root_path = get_project_root()
        
        grammar_file = (root_path / 'src/grammars' / f"{grammar_name}.ebnf").resolve()
        with grammar_file.open() as file:
            self.grammar_str = file.read()

        self.parsed_grammar = parse_ebnf(self.grammar_str)
        first_rule = self.grammar_str.split("\n")[0]
        #print(f"{grammar_name}: {first_rule}")

        
        
    def generate_text(self, prompt):
        try: 
            full_prompt = prompt
            if self.model_type == "instruct":
                messages = [
                    {
                        "role": "user", 
                        "content": "You are a helpful chemical assistant that only answers a valid SMILES string that represents a molecule based on the given information, if any is given.\n" + prompt}
                ]
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            input_ids = self.tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
            
            self.grammar = IncrementalGrammarConstraint(self.grammar_str, "root", self.tokenizer)
            self.grammar_processor = GrammarConstrainedLogitsProcessor(self.grammar)

            with torch.no_grad():
                constrained_output = self.model.generate(
                    **input_ids, 
                    max_new_tokens = self.max_new_tokens,
                    do_sample=False,
                    logits_processor=[self.grammar_processor],
                    repetition_penalty=self.repetition_penalty,
                    num_return_sequences=self.num_return_sequences)
            
            string_grammar = StringRecognizer(self.parsed_grammar.grammar_encoding, self.parsed_grammar.symbol_table["root"])
            prompt_length = len(self.tokenizer.decode(input_ids['input_ids'][0], skip_special_tokens=True))

            
            res = self.tokenizer.decode(
                constrained_output[0],
                skip_special_tokens=True,
            )

            return res[prompt_length:]
        except Exception as e:
            print(e)
 
    def debug_generate_text(self, prompt):
        try: 
            full_prompt = prompt
            if self.model_type == "instruct":
                messages = [
                    {
                        "role": "user", 
                        "content": "You are a helpful chemical assistant that only answers a valid SMILES string that represents a molecule based on the given information, if any is given.\n" + prompt}
                ]
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            input_ids = self.tokenizer([full_prompt], add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
            
            with torch.no_grad():
                unconstrained_output = self.model.generate(
                    **input_ids, 
                    do_sample=True,
                    temperature=1,
                    max_new_tokens = self.max_new_tokens)

                constrained_output = self.model.generate(
                    **input_ids, 
                    max_new_tokens = self.max_new_tokens,
                    do_sample=False,
                    logits_processor=[self.grammar_processor],
                    repetition_penalty=self.repetition_penalty,
                    num_return_sequences=self.num_return_sequences)
            
            string_grammar = StringRecognizer(self.parsed_grammar.grammar_encoding, self.parsed_grammar.symbol_table["root"])
            
            prompt_length = len(self.tokenizer.decode(input_ids['input_ids'][0], skip_special_tokens=True))

            
            res = self.tokenizer.decode(
                constrained_output[0],
                skip_special_tokens=True,
            )

            # decode output
            generations = self.tokenizer.batch_decode(
                torch.concat([unconstrained_output, constrained_output]),
                skip_special_tokens=True,
            )
            
            responses = {}
            for generation, gen_type in zip(generations, ["Unconstrained", "Constrained"]):
                print(gen_type + ":")
                print(generation)
                
                responses[gen_type] = generation[prompt_length:]
                
                assert string_grammar._accept_prefix(
                    res[prompt_length:]
                ), f"The generated prefix does not match the grammar: {string_grammar._accept_prefix(res[prompt_length:])}"
                print(
                    f"The generation matches the grammar: {string_grammar._accept_string(generation[prompt_length:])}"
                )

            return responses
        except Exception as e:
            print(e)
            
if __name__ == "__main__":
    model = MistralCFGModel(
        #model_id="mistralai/Mistral-7B-v0.1", 
        model_id="mistralai/Mistral-7B-Instruct-v0.2", 
        do_sample=False,
        max_new_tokens=20,
        repetition_penalty=1.9,
        num_return_sequences=1
    )
    messages = "Generate a simple molecule in SMILES format:"
    out = model.generate_text(messages)
    print(out)
    print("Is constrained valid?: ") 
    print(is_valid_smiles(out))