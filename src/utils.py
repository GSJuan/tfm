import pandas as pd
import os

def log_results(file, dataset_name, model_name, strategy, prompt_template, num_generations, input_samples, generated_response, validity, novelty, uniqueness, drug_likeness, sample_size=None):
    """Log the results."""
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "prompting_strategy": strategy,
        "prompt_template": prompt_template,
        "num_generations": num_generations,
        "input_samples": input_samples,
        "generated_response": generated_response,
        "validity": validity,
        "novelty": novelty,
        "uniqueness": uniqueness,
        "drug_likeness": drug_likeness
    }
    if sample_size is not None:
        result["sample_size"] = sample_size
        
    df = pd.DataFrame([result])
    df.to_csv(file, mode='a', header=not os.path.exists(file), index=False)
