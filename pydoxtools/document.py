from abc import ABC, abstractmethod

class Base(ABC):
    """
    This class is the base for all document classes in pydoxtools and
    defines a common interface for all.
    """

    @property
    @abstractmethod
    def tables(self):
        pass

