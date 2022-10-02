from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, width, heigth):
        self.width = width
        self.heigth = heigth
    @abstractmethod
    def area(self) -> float:
        pass

class Triangle(Shape):
    def area(self) -> float:
        return self.width * self.heigth / 2

class Rectangle(Shape):
    def area(self) -> float:
        return float(self.width * self.heigth)


