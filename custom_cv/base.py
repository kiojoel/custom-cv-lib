from abc import ABC, abstractmethod

class BaseCVSplitter(ABC):
    @abstractmethod
    def split(self, X, y=None, groups=None):
        pass


    def get_n_splits(self):
        return self.n_splits