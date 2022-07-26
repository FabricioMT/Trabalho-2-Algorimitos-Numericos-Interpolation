import sys
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from os import listdir
from math import sqrt

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
    A = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', usecols=(0,1))
    return A

def fatoraLU(A):

    U = np.copy(A)
    dimensionM = A.shape[0]
    L = np.eye(dimensionM)

    for j in np.arange(dimensionM-1):
        for i in np.arange(j+1,dimensionM):
            L[i,j] = U[i,j]/U[j,j]
            for k in np.arange(j+1,dimensionM):
                U[i,k] = U[i,k] - L[i,j]*U[j,k]
            U[i,j] = 0

    return L, U

def Uy(U, y):

    x = np.zeros_like(y)

    for i in range(len(x), 0, -1):
      x[i-1] = (y[i-1] - np.dot(U[i-1, i:], x[i:])) / U[i-1, i-1]

    return x

def lu_solve(L, U, b):
    y = Lb(L,b)
    x = Uy(U,y)

    return x

def Lb(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i, j]*y[j])
        y[i] = y[i]/L[i, i]

    return y

def LU(A,b):
    start = timer()
    L,U = fatoraLU(A)
    x = lu_solve(L,U,b)
    end = timer()

    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução LU: {timing}\n")
    return x

def cholesky(A):
    start = timer()

    dimensionM = A.shape[0]
    MI = np.zeros_like(A)

    for k in np.arange(dimensionM):
        MI[k,k] = sqrt(A[k,k])
        MI[k,k+1:] = A[k,k+1:]/MI[k,k]
        for j in np.arange(k+1,dimensionM):
            A[j,j:] = A[j,j:] - MI[k,j] * MI[k,j:]
    
    end = timer()

    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução Cholesky: {timing}\n")
    MI = np.diag(MI)
    return MI

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

def seidel(A,B,precision):  
    start = timer()
    A = A.astype(np.double)
    B = B.astype(np.double)

    dimensionM = A.shape[0]
    DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    x0 = DiagA/B
    x0 = np.diag(x0)
    x0 = x0.astype(np.double)
    x = np.copy(x0)

    D = precision + 1
    while (D > precision):  
        for i in np.arange(dimensionM):  
            x[i] = B[i]
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,dimensionM))):
                x[i] -= A[i,j]*x[j]
            x[i] /= A[i,i]
        d = np.linalg.norm(x-x0,np.inf)
        D = d/max(np.fabs(x))

        if (D < precision):
            end = timer()
            timing = timedelta(seconds=end-start)
            print(f"\ntempo de execução [Seidel]: {timing}\n")
            return x
        x0 = np.copy(x)


if __name__ == '__main__':
    inputs = readArgs()
    A = readFile(inputs)
    print(A)

