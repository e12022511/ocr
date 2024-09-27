from abc import ABC, abstractmethod


class DeskewMethod(ABC):
    """
       Abstract base class for implementing various deskewing methods.

       Any subclass that inherits from DeskewMethod must implement the deskew method.
       This structure enforces that all deskewing methods share a common interface.

       Methods:
           deskew(image): Abstract method for deskewing an image. Must be implemented by subclasses.

       """
    @abstractmethod
    def deskew(self, image):
        """
                Abstract method for deskewing the provided image.

                Args:
                    image (numpy.ndarray): The image to be deskewed, represented as a NumPy array.

                Returns:
                    numpy.ndarray: Deskewed image.

                This method must be implemented by any subclass.
                """
        pass
