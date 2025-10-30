from __future__ import annotations

from collections.abc import MutableSequence
from itertools import chain
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import TypeAlias
from typing import cast
from typing import overload

import numpy as np
import importlib
import time
import tracemalloc

N: TypeAlias = int | float
Row: TypeAlias = list[N]

class OperationCounter:
    additions = 0
    multiplications = 0
    
    @classmethod
    def reset(cls):
        cls.additions = 0
        cls.multiplications = 0
    
    @classmethod
    def report(cls):
        return f"Operations: {cls.multiplications} multiplications, {cls.additions} additions"

class Matrix(MutableSequence[Row]):
    def __init__(self, data: Iterable[Iterable[N]] | None = None) -> None:
        if data is None:
            self._data: list[Row] = []
        else:
            self._data = [list(row) for row in data]

    @overload
    def __getitem__(self, index: int) -> Row: ...
    @overload
    def __getitem__(self, index: slice) -> Matrix: ...

    def __getitem__(self, index: int | slice) -> Row | Matrix:
        if isinstance(index, slice):
            return Matrix(self._data[index])
        return self._data[index]

    @overload
    def __setitem__(self, index: int, value: Row) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[Row]) -> None: ...

    def __setitem__(self, index: int | slice, value: Row | Iterable[Row]) -> None:
        if isinstance(index, slice):
            rows = [list(row) for row in cast(Iterable[Row], value)]
            self._data[index] = rows
        else:
            if not isinstance(value, list):
                raise TypeError("single row assignment requires a list of int or float")
            self._data[index] = value

    def __delitem__(self, index: int | slice) -> None:
        del self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def insert(self, index: int, value: Row) -> None:
        self._data.insert(index, value)

    def __iter__(self) -> Iterator[Row]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Matrix({self._data!r})"
    
    def __repr__(self) -> str:
        rounded_str = '\n          '.join([str([f"{x:.5f}" for x in row]) for row in self._data])
        return rounded_str

    @classmethod
    def block(cls, blocks) -> Matrix:
        m = Matrix()
        for hblock in blocks:
            for row in zip(*hblock):
                m.append(list(chain.from_iterable(row)))

        return m

    def binet(self, b: Matrix) -> Matrix:
        assert self.shape.cols == b.shape.rows
        m = Matrix()
        for row in self:
            new_row = []
            for c in range(len(b[0])):
                col = [b[r][c] for r in range(len(b))]
                dot_product = 0
                for x, y in zip(row, col):
                    OperationCounter.multiplications += 1
                    product = x * y
                    if dot_product != 0 or product != 0:
                        OperationCounter.additions += 1
                    dot_product += product
                new_row.append(dot_product)
            m.append(new_row)
        return m
    
    def __matmul__(self, b: Matrix) -> Matrix:
        return self.binet(b)

    def __add__(self, b:Matrix) -> Matrix:
        return self.add(b, True)
    
    def __addnocount__(self, b:Matrix) -> Matrix:
        return self.add(b, False)
    
    def __neg__(self) -> Matrix:
        return Matrix([[-x for x in row] for row in self])

    def add(self, b: Matrix, flag: bool) -> Matrix:
        assert self.shape == b.shape
        rows, cols = self.shape
        result = Matrix()
        for i in range(rows):
            row = []
            for j in range(cols):
                if flag:
                    OperationCounter.additions += 1
                row.append(self[i][j] + b[i][j])
            result.append(row)
        return result

    def __sub__(self, b: Matrix) -> Matrix:
        assert self.shape == b.shape
        rows, cols = self.shape
        result = Matrix()
        for i in range(rows):
            row = []
            for j in range(cols):
                OperationCounter.additions += 1
                row.append(self[i][j] - b[i][j])
            result.append(row)
        return result
    
    def __eq__(self, other:Matrix) -> bool:
        rows, cols = self.shape
        if self.shape != other.shape:
            return False
        
        for i in range(rows):
            for j in range(cols):
                if abs(self[i][j] - other[i][j]) > 1e-9:
                    return False
        
        return True
    
    def copy(self):
        return type(self)(self._data)

    def round(self, ndigits: int | None = None) -> Matrix:
        return Matrix([[round(i, ndigits) for i in row] for row in self])

    @property
    def shape(self) -> Shape:
        cols = len(self[0]) if self else 0
        return Shape(len(self), cols)
    
    

    def inverse(self) -> Matrix:
        assert self.shape.cols == self.shape.rows
        def closest_power(n: int) -> int:
            n -= 1
            n |= n >> 1
            n |= n >> 2
            n |= n >> 4
            n |= n >> 8
            n |= n >> 16
            n += 1
            return n

        n = self.shape.rows
        if n == 1:
            assert self[0][0] != 0, "Can't inverse"
            OperationCounter.multiplications += 1
            return Matrix([[1/self[0][0]]])
        if n == 2:
            a, b = self[0]
            c, d = self[1]
            determ = a * d - b * c
            assert determ != 0, "Can't inverse"

            OperationCounter.multiplications += 2
            OperationCounter.additions += 1

            inv = Matrix([[d / determ, -b / determ],
                        [-c / determ, a / determ]])
            return inv
        
        m = closest_power(n)
        if m != n:
            padded = Matrix([row[:] + [0]*(m-n) for row in self])
            for _ in range(m-n):
                padded.append([0]*m)
            for i in range(n, m):
                padded[i][i] = 1
            padded_inv = padded.inverse()
            trimmed = Matrix([row[:n] for row in padded_inv[:n]])
            return trimmed
        
        p = n // 2

        a11 = Matrix([r[:p] for r in self[:p]])
        a12 = Matrix([r[p:] for r in self[:p]])
        a21 = Matrix([r[:p] for r in self[p:]])
        a22 = Matrix([r[p:] for r in self[p:]])

        a11_inv = a11.inverse()
        s22 = a22 - a21.binet(a11_inv).binet(a12)
        s22_inv = s22.inverse()

        b11 = a11_inv + a11_inv.binet(a12).binet(s22_inv).binet(a21).binet(a11_inv)
        b12 = -a11_inv.binet(a12).binet(s22_inv)    
        b21 = -s22_inv.binet(a21).binet(a11_inv)
        b22 = s22_inv

        return Matrix.block([[b11, b12], [b21, b22]])


    def gauss(self, b: Matrix) -> Matrix:
        assert self.shape.cols == self.shape.rows
        assert self.shape.rows == b.shape.rows
        n = self.shape.rows

        p = n // 2

        a11 = Matrix([r[:p] for r in self[:p]])
        a12 = Matrix([r[p:] for r in self[:p]])
        a21 = Matrix([r[:p] for r in self[p:]])
        a22 = Matrix([r[p:] for r in self[p:]])
        b1 = Matrix(r for r in b[:p])
        b2 = Matrix(r for r in b[p:])

        l11, u11 = self.lufac()
        l11_inv = l11.inverse()
        u11_inv = u11.inverse()

        s = a22 - a21.binet(u11_inv).binet(l11_inv).binet(a12)
        ls, us = s.lufac()
        
        c11 = u11
        c12 = l11_inv.binet(a12)
        c22 = us

        rhs1 = l11_inv.binet(b1)
        ls_inv = ls.inverse()
        rhs22 = ls_inv.binet(b2) - ls_inv.binet(a21).binet(u11_inv).binet(l11_inv).binet(b1)


    def lufac(self) -> tuple[Matrix, Matrix]:
        rows, cols = self.shape
        if rows != cols: raise "Macierz musi być kwadratowa"
        
        return self.recursive_lufac()

    def recursive_lufac(self):
        n = self.shape.rows
        if n == 1:
            return Matrix([[0]]), self

        if n == 2:
            a11 = self[0][0]
            a12 = self[0][1]
            a21 = self[1][0]
            a22 = self[1][1]
            L = id(2)
            L[1][0] = a21 / a11

            u11 = a11
            u12 = a12
            u22 = a22 - a21 * a12 / a11

            return L, Matrix([[u11, u12], [0, u22]])
        
        p = n // 2

        a11 = Matrix([r[:p] for r in self[:p]])
        a12 = Matrix([r[p:] for r in self[:p]])
        a21 = Matrix([r[:p] for r in self[p:]])
        a22 = Matrix([r[p:] for r in self[p:]])
        
        l11, u11 = a11.recursive_lufac()
        u11i = u11.inverse()
        l21 = a21.binet(u11i)
        l11i = l11.inverse()
        u12 = l11i.binet(a12)
        l22 = a22 - a21.binet(u11i).binet(l11i).binet(a12)
        l22, u22  = l22.recursive_lufac()

        L = Matrix.block([[l11, zeros(p)], [l21, l22]])
        U = Matrix.block([[u11, u12], [zeros(p), u22]])

        return L, U

    def det(self) -> N:
        _, U = self.lufac()
        result = 1
        for i in range(U.shape.rows):
            result *= U[i][i]
        return result
        
def zeros(n:int) -> Matrix:
    return Matrix([[0] * n for _ in range(n)])

def id(n: int) -> Matrix:
    M = zeros(n)
    for i in range(n):
        M[i][i] = 1
    return M


class Shape(NamedTuple):
    rows: int
    cols: int

def biggest():
    i = 2
    while True:
        a = Matrix(np.random.rand(i, i))
        b = Matrix(np.random.rand(i, i))

        OperationCounter.reset()   
        start = time.monotonic()
        ai = a.inverse()
        end = time.monotonic()
        inverse_additions = OperationCounter.additions
        inverse_multiplications = OperationCounter.multiplications
        inverse_time = end - start

        OperationCounter.reset()   
        start = time.monotonic()
        l, u = a.lufac()
        assert l @ u == a
        end = time.monotonic()
        lufac_additions = OperationCounter.additions
        lufac_multiplications = OperationCounter.multiplications
        lufac_time = end - start
        
        print(i,
              inverse_additions, inverse_multiplications, inverse_time,
              lufac_additions, lufac_multiplications, lufac_time
              )
        i *= 2

biggest()
'''
m1 = Matrix([[1, 1, 1, 0], 
             [0, 3, 1, 2],
             [2, 3, 1, 0],
             [1, 0, 2, 1]])
print(m1.inverse())

for i in range(2, 10):
    m = Matrix(np.random.rand(2 ** i, 2 ** i))
    l, u = m.lufac()
    if l.binet(u) == m:
        print(str(i) + " OK")
    else:
        print("ZLE")
        print(m)
        print(l)
        print(u)
'''