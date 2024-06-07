from abc import ABC, abstractmethod
from os.path import exists
import pandas as pd
from random import sample 

class Reader(ABC):
    
    @abstractmethod
    def load(self, path):
        pass

class MosesReader(Reader):
    path = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
    def __init__(self):
        self.df = None
        self.__load()
    
    def __load(self, path = None):
        if path == None:
            self.df = pd.read_csv(MosesReader.path, usecols=['SMILES']).squeeze("columns").astype(str).tolist()
            return self.df
        else : return None
    
    def load(self, path):
        if exists(path):
            return self.__load(path)
        else: 
            print("Incorrect path, file doesn't exist!")
            
    def get_data(self):
        if self.df is None:
            raise ValueError("No data loaded")
        return self.df
    
    def get_sample(self, sample_size):
        if self.df is None:
            raise ValueError("No data loaded")
        return sample(self.df, sample_size)
            
if __name__ == '__main__':
    moses = MosesReader()
    print(moses.get_data()[0])

    