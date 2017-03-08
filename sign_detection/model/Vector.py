

class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    @property
    def as_array(self):
        return [self.x, self.y]

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __radd__(self, other):
        return Vector(self.x + other, self.y + other)

    def __rsub__(self, other):
        return Vector(self.x - other, self.y - other)

    def __str__(self):
        return str([self.x, self.y])


def from_array(arr):
    return Vector(arr[0], arr[1])

