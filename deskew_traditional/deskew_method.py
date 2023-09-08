from abc import ABC, abstractmethod


class DeskewMethod(ABC):
    @abstractmethod
    def deskew(self, image):
        pass

