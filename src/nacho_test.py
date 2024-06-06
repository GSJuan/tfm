from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from rdkit.Chem import MolFromSmiles
import string
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


atoms_tokens = ['Ag','Al','As','Au','B','Ba','Bi','Br','C','Ca',
              'Cd','Cl','Co','Cr','Cs','Cu','F','Fe','Ga','Gd',
              'Ge','H','Hg','I','In','K','Li','M','Mg','Mn',
              'Mo','N','Na','O','P','Pt','Ru','S','Sb','Sc',
              'Se','Si','Sn','V','W','Z','Zn','c','e','n','o','p','s']

atoms_tokens = sorted(atoms_tokens, key=lambda s: len(s), reverse=True)
SMI_REGEX_PATTERN = r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|" + '|'.join(atoms_tokens) + ")"
regex = re.compile(SMI_REGEX_PATTERN)


def clean_output_sequence(output_sequence):
    return output_sequence.replace('</s>', '').replace('<sm_', '').replace(' sm_', '').replace('>', '').strip()


def add_special_symbols(text):
  output = []
  for word in text.split():
      tokens = [token for token in regex.findall(word)]
      if len(tokens) > 4 and (word == ''.join(tokens)) and MolFromSmiles(word):
          output.append(''.join(['<sm_'+t+'>' for t in tokens]))
      else:
          output.append(word)
  return ' '.join(output)


PROMPT = """Given the following reactants and reagents, please provide a possible product. 
          CCN(CC)CC.CCN=C=NCCCN(C)C.CN(C)C=O.Cl.NC1=CC=C(Cl)C=C1N.O.O=C(O)CCCCCNC(=O)C=C1C2=CC=CC=C2C2=CC=CC=C12.OC1=CC=CC2=C1N=NN2.[Cl-].[Na+]"""
PROMPT = add_special_symbols(PROMPT)

model = AutoModelForSeq2SeqLM.from_pretrained('insilicomedicine/nach0_base')
tokenizer = AutoTokenizer.from_pretrained('insilicomedicine/nach0_base')

input_text_ids = tokenizer(PROMPT, padding="longest", max_length=512, truncation=True, return_tensors="pt")
generated_text_ids = model.generate(**input_text_ids, do_sample=True, top_k=100, top_p=0.95, max_length=512)
generated_text = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)[0]
generated_text = clean_output_sequence(generated_text)
print(generated_text)