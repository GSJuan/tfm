from syncode import Syncode
import torch
import re


class BaseConstrainedModel:
    def __init__(self, temperature=0.0, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.7):
        # Grammar credits: https://depth-first.com/articles/2020/04/20/smiles-formal-grammar/
        self.grammar = r"""
start: line

line: atom (chain | branch)*
chain: (dot atom | bond? (atom | rnum))+
branch: "(" ((bond | dot)? line)+ ")"
atom: organic_symbol | bracket_atom
bracket_atom: "[" isotope? symbol chiral? hcount? charge? map? "]"
rnum: digit | "%" digit digit
isotope: digit? digit? digit
hcount: "H" digit?
charge: "+" ("+" | fifteen)? | "-" ("-" | fifteen)?
map: ":" digit? digit? digit
symbol: "A" ("c" | "g" | "l" | "m" | "r" | "s" | "t" | "u")
       | "B" ("a" | "e" | "h" | "i" | "k" | "r")?
       | "C" ("a" | "d" | "e" | "f" | "l" | "m" | "n" | "o" | "r" | "s" | "u")?
       | "D" ("b" | "s" | "y")
       | "E" ("r" | "s" | "u")
       | "F" ("e" | "l" | "m" | "r")?
       | "G" ("a" | "d" | "e")
       | "H" ("e" | "f" | "g" | "o" | "s")?
       | "I" ("n" | "r")?
       | "K" "r"?
       | "L" ("a" | "i" | "r" | "u" | "v")
       | "M" ("c" | "g" | "n" | "o" | "t")
       | "N" ("a" | "b" | "d" | "e" | "h" | "i" | "o" | "p")?
       | "O" ("g" | "s")?
       | "P" ("a" | "b" | "d" | "m" | "o" | "r" | "t" | "u")?
       | "R" ("a" | "b" | "e" | "f" | "g" | "h" | "n" | "u")
       | "S" ("b" | "c" | "e" | "g" | "i" | "m" | "n" | "r")?
       | "T" ("a" | "b" | "c" | "e" | "h" | "i" | "l" | "m" | "s")
       | "U" | "V" | "W" | "Xe" | "Y" "b"?
       | "Z" ("n" | "r")
       | "b" | "c" | "n" | "o" | "p" | "s" "e"? | "as"
organic_symbol: "B" "r"? | "C" "l"? | "N" | "O" | "P" | "S"
               | "F" | "I" | "At" | "Ts"
               | "b" | "c" | "n" | "o" | "p" | "s"
bond: "-" | "=" | "#" | "$" | "/" | "\\"
dot: "."
chiral: "@"? "@"
digit: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
fifteen: "1" ("0" | "1" | "2" | "3" | "4" | "5")? | ("2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")

%ignore " "
"""

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

            
class MistralConstrainedModel(BaseConstrainedModel):
   
    def __init__(self, model_id, temperature=1.0, max_new_tokens=20, do_sample=True, top_k=50, top_p=0.7):
        
        super().__init__(temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p)
        try: 
            self.model = Syncode(model=model_id, quantize=False, mode="grammar_strict", grammar=self.grammar, chat_mode=False, parser='lalr', parse_output_only=True, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p)
            
        except Exception as e:
            print(e)
        
    def generate_text(self, prompt):
        try:
            
            
            response = self.model.infer(prompt)
            return response
        except Exception as e:
            print(e)

if __name__ == "__main__":
    model = MistralConstrainedModel(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=1,
        max_new_tokens=356,
    )
    messages = "Generate a novel molecule in SMILES format.\nAnswer:"
    out = model.generate_text(messages)
    print(out)