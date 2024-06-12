from abc import ABC, abstractmethod
from os.path import exists
import pandas as pd
from random import sample 
import json


class BaseReader:
    def __init__(self, source, config):
        self.source = source
        self.config = config

    def download_data(self):
        """Download data from the source if necessary."""
        raise NotImplementedError("Subclasses should implement this method")

    def extract_smiles(self):
        """Extract SMILES strings from the data."""
        raise NotImplementedError("Subclasses should implement this method")

        
class CSVReader(BaseReader):
    def __init__(self, source, config):
        super().__init__(source, config)
        self.smiles_column = config.get('smiles_column', 'smiles')
        self.split_column = config.get('split_column', 'split')
        self.split_value = config.get('split', 'both') # can be 'train', 'test', or 'both'

    def download_data(self):
        """Assume the data is local or already downloaded."""
        pass

    def extract_smiles(self):
        data = pd.read_csv(self.source, usecols=[self.smiles_column, self.split_column])

        if self.split_value != 'both':
            data = data[data[self.split_column] == self.split_value]

        return data[self.smiles_column].astype(str).tolist()
            
class JSONReader(BaseReader):
    def __init__(self, source, config):
        super().__init__(source, config)
        self.smiles_key = config.get('smiles_key', 'smiles')

    def download_data(self):
        """Assume the data is local or already downloaded."""
        pass

    def extract_smiles(self):
        with open(self.source, 'r') as file:
            data = json.load(file)
        return [entry[self.smiles_key] for entry in data]

    
def main():
    datasets = {
        "Moses": {
            "reader": CSVReader,
            "source": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
            "config": {"smiles_column": "SMILES",
                       "split_column": "split",
                       "split_value": "both"}
        }
    }
    all_smiles = {}
    for name, dataset in datasets.items():
        reader_class = dataset["reader"]
        source = dataset["source"]
        config = dataset["config"]
        reader = reader_class(source, config)
        smiles = reader.extract_smiles()
        all_smiles[name+"_"+config["split_value"]] = smiles

# Example processing of the extracted SMILES strings
    for dataset_name, smiles_list in all_smiles.items():
        print(f"{dataset_name} contains {len(smiles_list)} SMILES strings.")

if __name__ == "__main__":
    main()