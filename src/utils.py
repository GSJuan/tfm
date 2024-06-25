import pandas as pd
import os
from pathlib import Path
import re

def clean_error_message(error_message):
    
    # Check if the object is not an instance of str
    if not isinstance(error_message, str):
        # Check if the object has a __str__ method
        if hasattr(error_message, '__str__'):
            error_message = error_message.__str__()  # Explicitly call the __str__ method
        elif hasattr(error_message, '__repr__'):
            error_message = error_message.__repr__()  # Explicitly call the __str__ method
        else: 
            error_message = str(error_message)
    # Use regular expressions to match and remove the leading time if it exists
    cleaned_message = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", error_message)
    # Remove the trailing newline character if it exists
    cleaned_message = cleaned_message.rstrip('\n')
    return cleaned_message

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def log_results(file, dataset_name, model_name, strategy, prompt_template, num_generations, input_samples, generated_response, validity_metrics, novelty, uniqueness, drug_likeness, sample_size=None):
    try:
        """Log the results."""
        result = {
            "dataset": dataset_name,
            "model": model_name,
            "prompting_strategy": strategy,
            "prompt_template": prompt_template,
            "input_samples": input_samples,
            "num_generations": num_generations,
            "generated_responses": generated_response,
            "novelty": novelty,
            "uniqueness": uniqueness,
            "drug_likeness": drug_likeness
        }
        for key, value in validity_metrics.items():
            result[key] = value
            
        if sample_size is not None:
            result["few_shot_sample_size"] = sample_size

        df = pd.DataFrame([result])
        df.to_csv(file, mode='a', header=not os.path.exists(file), index=False)
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Example usage
    error_message = "[08:49:46] SMILES Parse Error: unclosed ring for input: 'I.InspoF:CC1=C(O)Nc2cccncs'\n"
    cleaned_message = clean_error_message(error_message)
    print(cleaned_message)