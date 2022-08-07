import sys
import numpy as np
from math import sqrt
from os import listdir
from datetime import timedelta
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from typing import Tuple, List
import bisect
from scipy.interpolate import CubicSpline, interp1d

def readArgs():
    if len(sys.argv) == 2:
        arq = 0
        file_input = listdir('./inputs/')
        for file in file_input:
            if file == sys.argv[1]:
                arq = 'inputs/' + sys.argv[1]
        
        args = sys.argv[1:]
        print(f"Arguments count: {len(sys.argv)}")
        print(f"Arguments of the script : {args}")
        if arq == 0:
            print("Arquivo não encontrado nos Inputs ou Arquivo com nome Inválido !")
            exit(1) 
    else:
        print("Entrada de Dados Inválida !")
        exit(1)
    return arq

def readFile(inputs):
    x = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', usecols=(0))
    y = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', usecols=(1))
    return x, y

def jacobi(A,B,precision):
    start = timer()
    dimensionM = A.shape[0]
    x = np.zeros(dimensionM)

    DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    x0 = DiagA/B
    x0 = np.diag(x0)
    x0 = x0.astype(np.double)

    D = precision + 1
    while (D > precision):  
        for i in np.arange(dimensionM):  
            x[i] = B[i]
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,dimensionM))):
                x[i] -= A[i,j]*x0[j]
            x[i] /= A[i,i]

        d = np.linalg.norm(x-x0,np.inf)
        D = d/max(np.fabs(x))

        if (D < precision):
            end = timer()
            timing = timedelta(seconds=end-start)
            print(f"\nTempo de execução [Jacobi]: {timing}\n")
            return x
        x0 = np.copy(x)

def compute_changes(x: List[float]) -> List[float]: 
    return [x[i+1] - x[i] for i in range(len(x) - 1)]

def create_tridiagonalmatrix(n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
    A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
    B = [2] * n
    C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
    return A, B, C

def create_target(n: int, h: List[float], y: List[float]):
    return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + [0]

def solve_tridiagonalsystem(A: List[float], B: List[float], C: List[float], D: List[float]):
    c_p = C + [0]
    d_p = [0] * len(B)
    X = [0] * len(B)

    c_p[0] = C[0] / B[0]
    d_p[0] = D[0] / B[0]
    for i in range(1, len(B)):
        c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
        d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / (B[i] - c_p[i - 1] * A[i - 1])

    X[-1] = d_p[-1]
    for i in range(len(B) - 2, -1, -1):
        X[i] = d_p[i] - c_p[i] * X[i + 1]

    return X

def compute_spline(x: List[float], y: List[float]):
    n = len(x)
    if n < 3:
        raise ValueError('Too short an array')
    if n != len(y):
        raise ValueError('Array lengths are different')

    h = compute_changes(x)
    if any(v < 0 for v in h):
        raise ValueError('X must be strictly increasing')

    A, B, C = create_tridiagonalmatrix(n, h)
    D = create_target(n, h, y)

    M = solve_tridiagonalsystem(A, B, C, D)

    coefficients = [[(M[i+1]-M[i])*h[i]*h[i]/6, M[i]*h[i]*h[i]/2, (y[i+1] - y[i] - (M[i+1]+2*M[i])*h[i]*h[i]/6), y[i]] for i in range(n-1)]

    def spline(val):
        idx = min(bisect.bisect(x, val)-1, n-2)
        z = (val - x[idx]) / h[idx]
        C = coefficients[idx]
        return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

    return spline

if __name__ == '__main__':
    inputs = readArgs()
    x, y = readFile(inputs)

#plot
    fig, ax = plt.subplots()

    ax.plot(x, y,'.-')

    ax.set(xlim=(0, 10), xticks=np.arange(0, 11),
            ylim=(0, 10), yticks=np.arange(0, 11))
    plt.show()  
       
    spline = compute_spline(x, y)

    for i, k in enumerate(x):
        assert abs(y[i] - spline(k)) < 1e-8, f'Error at {k}, {y[i]}'

    x_vals = [v / 10 for v in range(0, 50, 1)]
    y_vals = [spline(j) for j in x_vals]

    plt.plot(x_vals, y_vals)
    plt.show()

