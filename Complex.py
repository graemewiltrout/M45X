"""
Numerical Methods Package: Complex Numbers and Operations
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __repr__(self):
        return f"{self.real} + {self.imag}i"

    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.real + other.real, self.imag + other.imag)
        else:
            return Complex(self.real + other, self.imag)

    def __sub__(self, other):
        if isinstance(other, Complex):
            return Complex(self.real - other.real, self.imag - other.imag)
        else:
            return Complex(self.real - other, self.imag)

    def __mul__(self, other):
        if isinstance(other, Complex):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return Complex(real, imag)
        else:
            return Complex(self.real * other, self.imag * other)

    def __truediv__(self, other):
        if isinstance(other, Complex):
            denominator = other.real ** 2 + other.imag ** 2
            real = (self.real * other.real + self.imag * other.imag) / denominator
            imag = (self.imag * other.real - self.real * other.imag) / denominator
            return Complex(real, imag)
        else:
            return Complex(self.real / other, self.imag / other)

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def conjugate(self):
        return Complex(self.real, -self.imag)
