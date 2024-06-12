def log_results(results, dataset_name, model_name, strategy, prompt_template, num_generations, validity, novelty, uniqueness, drug_likeness, sample_size=None):
    """Log the results."""
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "prompting_strategy": strategy,
        "prompt_template": prompt_template,
        "num_generations": num_generations,
        "validity": validity,
        "novelty": novelty,
        "uniqueness": uniqueness,
        "drug_likeness": drug_likeness
    }
    if sample_size is not None:
        result["sample_size"] = sample_size
    results.append(result)
