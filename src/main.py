from models import Nach0GenerationModel, MistralGenerationModel
from gc_models import MistralConstrainedModel
from cfg_models import MistralCFGModel
from metrics import is_valid_smiles, evaluate
from readers import CSVReader, JSONReader, SMILESReader, SMIReader
from utils import log_results 
from random import sample, choice
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os

# Define configurations for each dataset
datasets = {
    "Moses": {
        "reader": CSVReader,
        "source": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
        "config": {"smiles_column": "SMILES",
                   "split_column": "SPLIT",
                   "split_value": "both"}
    },
    "Guacamole": {
        "reader": SMILESReader,
        "source": "/home/jovyan/tfm/data/guacamol_v1_all.smiles",
        "config": {}
    },
    "ZINC": {
        "reader": CSVReader,
        "source": "/home/jovyan/tfm/data/250k_rndm_zinc_drugs_clean_3.csv",
        "config": {"smiles_column": "smiles"}
    },
    "GDB13_Random": {
        "reader": SMIReader,
        "source": "/home/jovyan/tfm/data/gdb13.1M.freq.ll.smi",
        "config": {"smiles_column": "0"}
    }
}


models = {
    "constrained_mistral_base": {
        "class": MistralCFGModel,
        "model_id": "mistralai/Mistral-7B-v0.1"
    },
    "constrained_mistral_instruct": {
        "class": MistralCFGModel,
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "constrained_mixtral": {
        "class": MistralCFGModel,
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "args": {
            "load_in_4bit" = True
        }
    },
    "mistral_base": {
        "class": MistralGenerationModel,
        "model_id": "mistralai/Mistral-7B-v0.1"
    },
    "mistral_instruct": {
        "class": MistralGenerationModel,
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "mixtral": {
        "class": MistralGenerationModel,
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "args": {
            "load_in_4bit" = True
        }
    },
    "nach0": {
        "class": Nach0GenerationModel,
        "model_id": "insilicomedicine/nach0_base"
    }
} 

# Define prompting strategies
prompting_strategies = {
    "zero_shot": [
        "Create a new, never seen drug-like molecule and provide its SMILES string. Dont respond with anything apart from the SMILES string that encodes the molecule:",
        "Generate a novel molecule in SMILES format. Answer only the SMILES string: ",
    ],
    "one_shot": [
        "Generate a molecule in SMILES format similar to this one: [example_SMILES]",
        "",
    ],
    "few_shot": [
        """Here you have a sample of dataset, which contains SMILES strings that describe molecules. The task you have to accomplish is generate a novel molecule based on the inputs as possible. Answer only the SMILES string. 

MOLECULES:
[example_SMILES]


ANSWER:""",
        "Here are some SMILES strings of molecules: [example_SMILES]. Generate a similar molecule: "
    ]
}

# Define few-shot sample sizes
few_shot_sample_sizes = [3, 5, 10]

# Define number of SMILES to be generated for each prompt by each model
num_generations = 3


# Save results to CSV with a timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"final_molecule_generation_results_{timestamp}.csv"

# Main loop to process datasets, models, and prompting strategies
for model_name, model_config in models.items():

    model_class = model_config["class"]
    model_id = model_config["model_id"]
    
    if "args" in model_config:
        model = model_class(model_id, **model_config["args"])
    else:
        model = model_class(model_id)
    
    for dataset_name, dataset in datasets.items():
        reader_class = dataset["reader"]
        source = dataset["source"]
        config = dataset["config"]

        reader = reader_class(source, config)
        smiles_list = reader.extract_smiles()
        
        for strategy, prompt_templates in prompting_strategies.items():
            if strategy == "zero_shot":
                input_samples = ""
                for prompt in prompt_templates:
                    generated_smiles_list = []
                    for _ in tqdm(range(num_generations), desc=f"Processing {dataset_name} with {model_name} ({strategy})"):
                        generated_smiles = model.generate_text(prompt)
                        generated_smiles_list.append(generated_smiles)
                    validity_metrics, novelty, uniqueness, drug_likeness = evaluate(generated_smiles_list)
                    log_results(output_file, dataset_name, model_name, strategy, prompt, num_generations, input_samples, generated_smiles_list, validity_metrics, novelty, uniqueness, drug_likeness)
            
            elif strategy == "one_shot":
                for prompt_template in prompt_templates:
                    generated_smiles_list = []
                    sampled_smiles_list = []
                    for _ in tqdm(range(num_generations), desc=f"Processing {dataset_name} with {model_name} ({strategy})"):
                        smile = choice(smiles_list)
                        sampled_smiles_list.append(smile)
                        
                        prompt = prompt_template.replace("[example_SMILES]", smile)
                        
                        generated_smiles = model.generate_text(prompt)
                        generated_smiles_list.append(generated_smiles)
                        
                    validity_metrics, novelty, uniqueness, drug_likeness = evaluate(generated_smiles_list, sampled_smiles_list)
                    log_results(output_file, dataset_name, model_name, strategy, prompt_template, num_generations, sampled_smiles_list, generated_smiles_list, validity_metrics, novelty, uniqueness, drug_likeness)
                    
            elif strategy == "few_shot":
                for prompt_template in prompt_templates:
                    for sample_size in few_shot_sample_sizes: 
                        generated_smiles_list = []
                        sampled_smiles_list = []
                        for i in tqdm(range(num_generations), desc=f"Processing {dataset_name} with {model_name} ({strategy}, sample size {sample_size})"):
                            
                            subset_smiles = sample(smiles_list, sample_size)
                            sampled_smiles_list.extend(subset_smiles)
                            
                            example_smiles = "\n".join(subset_smiles)
                    
                            prompt = prompt_template.replace("[example_SMILES]", example_smiles)
                        
                            generated_smiles = model.generate_text(prompt)
                            generated_smiles_list.append(generated_smiles)
                            
                        validity_metrics, novelty, uniqueness, drug_likeness = evaluate(generated_smiles_list, subset_smiles)
                        log_results(output_file, dataset_name, model_name, strategy, prompt_template, num_generations, sampled_smiles_list, generated_smiles_list, validity_metrics, novelty, uniqueness, drug_likeness, sample_size)
    del model
    torch.cuda.empty_cache()

print(f"Results saved to {output_file}")