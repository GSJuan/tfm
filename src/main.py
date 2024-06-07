from readers import MosesReader
from models import Nach0GenerationModel, MixtralGenerationModel
from validate import validate_single_smiles


moses_smiles = MosesReader()
model = Nach0GenerationModel(
        model_id="insilicomedicine/nach0_base",
        temperature=1.0,
        max_new_tokens=356,
    )

molecules_sample = '\n'.join(str(i) for i in moses_smiles.get_sample(50))

prompt = f"""Here you have a set of MOSES dataset, which contains SMILES strings that describe molecules. The task you have to accomplish is generate a novel molecule based on the inputs as possible. Answer only the SMILES strings separated by a \n character. 

MOLECULES:
{molecules_sample}


ANSWER:"""

answer = model(prompt)

print("Clean output: " + answer)

validate_single_smiles(answer)