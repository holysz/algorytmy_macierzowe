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
    
    def recursive_binet(self, b:Matrix) -> Matrix:
        a = self
        rows, cols = a.shape
        assert rows == cols, "matrices must be square"
        assert a.shape == b.shape, "matrices must be the same shape"
        assert rows and (rows & rows - 1) == 0, "shape must be a power of 2"

        if rows == 1:
            return a.binet(b)

        p = rows // 2 
        
        a11 = Matrix([n[:p] for n in self[:p]])
        a12 = Matrix([n[p:] for n in self[:p]])
        a21 = Matrix([n[:p] for n in self[p:]])
        a22 = Matrix([n[p:] for n in self[p:]])

        b11 = Matrix([n[:p] for n in b[:p]])
        b12 = Matrix([n[p:] for n in b[:p]])
        b21 = Matrix([n[:p] for n in b[p:]])
        b22 = Matrix([n[p:] for n in b[p:]])

        c11 = (a11.recursive_binet(b11)).__addnocount__(a12.recursive_binet(b21))
        c12 = (a11.recursive_binet(b12)).__addnocount__(a12.recursive_binet(b22))
        c21 = (a21.recursive_binet(b11)).__addnocount__(a22.recursive_binet(b21))
        c22 = (a21.recursive_binet(b12)).__addnocount__(a22.recursive_binet(b22))

        return Matrix.block([[c11, c12], [c21, c22]])
    
    def ai(self, B:Matrix) -> Matrix:
        A = self
        assert A.shape[1] == B.shape[0]

        package_name = ("generated_multiplications.m"
                        + "_".join([str(A.shape[0]),
                                    str(B.shape[0]),
                                    str(B.shape[1])])
                        + "_generated")
        module = importlib.import_module(package_name)
        
        result, multiplications = module.multiply(A, B)
        OperationCounter.multiplications = multiplications
        return result
    
    def __matmul__(self, b: Matrix) -> Matrix:
        return self.binet(b)

    def __add__(self, b:Matrix) -> Matrix:
        return self.add(b, True)
    
    def __addnocount__(self, b:Matrix) -> Matrix:
        return self.add(b, False)

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
    

    def strassen(self, b: Matrix) -> Matrix:
        rows, cols = self.shape

        assert rows == cols, "matrices must be square"
        assert self.shape == b.shape, "matrices must be the same shape"
        assert rows and (rows & rows - 1) == 0, "shape must be a power of 2"

        if rows == 1:
            return self.binet(b)

        p = rows // 2

        a11 = Matrix([n[:p] for n in self[:p]])
        a12 = Matrix([n[p:] for n in self[:p]])
        a21 = Matrix([n[:p] for n in self[p:]])
        a22 = Matrix([n[p:] for n in self[p:]])

        b11 = Matrix([n[:p] for n in b[:p]])
        b12 = Matrix([n[p:] for n in b[:p]])
        b21 = Matrix([n[:p] for n in b[p:]])
        b22 = Matrix([n[p:] for n in b[p:]])

        m1 = (a11 + a22).strassen(b11 + b22)
        m2 = (a21 + a22).strassen(b11)
        m3 = a11.strassen(b12 - b22)
        m4 = a22.strassen(b21 - b11)
        m5 = (a11 + a12).strassen(b22)
        m6 = (a21 - a11).strassen(b11 + b12)
        m7 = (a12 - a22).strassen(b21 + b22)

        c11 = m1 + m4 - m5 + m7
        c12 = m3 + m5
        c21 = m2 + m4
        c22 = m1 - m2 + m3 + m6

        return Matrix.block([[c11, c12], [c21, c22]])

    def round(self, ndigits: int | None = None) -> Matrix:
        return Matrix([[round(i, ndigits) for i in row] for row in self])

    @property
    def shape(self) -> Shape:
        cols = len(self[0]) if self else 0
        return Shape(len(self), cols)


class Shape(NamedTuple):
    rows: int
    cols: int


def examples():
    a = Matrix(np.random.rand(4, 4))
    b = Matrix(np.random.rand(4, 4))
    c = Matrix(np.random.rand(8, 8))
    d = Matrix(np.random.rand(8, 8))
    e = Matrix(np.random.rand(16, 16))
    f = Matrix(np.random.rand(16, 16))

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)

    print("Naive matrix multiplication:")
    
    OperationCounter.reset()
    result = a @ b
    print(f"  a * b = {result}")
    print(f"  {OperationCounter.report()}")
    
    OperationCounter.reset()
    result = (c @ d)
    print(f"  c * d = {result}")
    print(f"  {OperationCounter.report()}")
    
    OperationCounter.reset()
    result = e @ f
    print(f"  e * f = {result}")
    print(f"  {OperationCounter.report()}")

    print("\nStrassen's matrix multiplication:")
    
    OperationCounter.reset()
    result = a.strassen(b)
    print(f"  a * b = {result}")
    print(f"  {OperationCounter.report()}")
    
    OperationCounter.reset()
    result = c.strassen(d)
    print(f"  c * d = {result}")
    print(f"  {OperationCounter.report()}")
    
    OperationCounter.reset()
    result = e.strassen(f)
    print(f"  e * f = {result}")
    print(f"  {OperationCounter.report()}")

def biggest():
    i = 2
    while True:
        a = Matrix(np.random.rand(i, i))
        b = Matrix(np.random.rand(i, i))

        OperationCounter.reset()   
        start = time.monotonic()
        tracemalloc.start()
        binet_result = a @ b
        _, binet_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.monotonic()
        binet_additions = OperationCounter.additions
        binet_multiplications = OperationCounter.multiplications
        binet_time = end - start

        OperationCounter.reset()   
        start = time.monotonic()
        tracemalloc.start()
        recursive_binet_result = a.recursive_binet(b)
        _, recursive_binet_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.monotonic()
        recursive_binet_additions = OperationCounter.additions
        recursive_binet_multiplications = OperationCounter.multiplications
        recursive_binet_time = end - start
        
        OperationCounter.reset()
        start = time.monotonic()
        tracemalloc.start()
        strassen_result = a.strassen(b)
        _, strassen_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.monotonic()
        strassen_additions = OperationCounter.additions
        strassen_multiplications = OperationCounter.multiplications
        strassen_time = end - start

        if i < 5:
            start = time.monotonic()
            ai_result = a.ai(b)
            end = time.monotonic()
            ai_time = end - start
            
            assert binet_result == ai_result, (binet_result, ai_result)

        assert binet_result == recursive_binet_result == strassen_result

        print(i,
              binet_additions, binet_multiplications, binet_time, binet_mem,
              recursive_binet_additions, recursive_binet_multiplications, recursive_binet_time, recursive_binet_mem,
              strassen_additions, strassen_multiplications, strassen_time, strassen_mem,
              )
        i *= 2

def ai_test():
    from generated_multiplications.shapes import shapes
    for shape in shapes:
        u, v, w = shape
        a = Matrix(np.random.rand(u, v))
        b = Matrix(np.random.rand(v, w))
        OperationCounter.reset()
        start = time.monotonic()
        c = a.ai(b)
        end = time.monotonic()
        print(shape, end - start, OperationCounter.multiplications)


if __name__ == "__main__":
    biggest()