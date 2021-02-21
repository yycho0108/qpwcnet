#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple
import tensorflow as tf
from functools import partial


class TripletDataset(ABC):

    @abstractmethod
    def __iter__(self) -> Tuple[str, str, str]:
        """
        Return a triplet of filenames corresponding to sequential frames.
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key: str) -> Tuple[str, str, str]:
        """ Get a triplet corresponding to a key """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """ Length of dataset """
        raise NotImplementedError()

    @abstractproperty
    def keys(self) -> List[str]:
        """ RAI keys """
        raise NotImplementedError()
