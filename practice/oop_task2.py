class Mother():
    name = "mother"
    
    def __str__(self):
        return self.name


class Daughter(Mother):
    def __init__(self):
        self.name = "daughter"

mother1, mother2 = Mother(), Mother()
mother2.name = "mother2"
print(mother1, mother2)

daughter = Daughter()
print(daughter)
