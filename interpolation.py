import sys
import bisect
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def readArgs():
    if len(sys.argv) >= 2:
        arq = 0
        amplitude = float(sys.argv[2])
        file_input = listdir('./inputs/')
        
        if amplitude > 2:
            pass
        else:
            print("Amplitude muito baixa !")
            amplitude = 2
            print("Amplitude set default = 2.")
        
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
    return arq, amplitude

def readFile(inputs):
    x = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', usecols=(0))
    y = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', usecols=(1))
    return x, y

def CalcH(x): return [x[i+1] - x[i] for i in range(len(x) - 1)]

def CalcD(n, h, y):
    d0 = [0]
    dn = [0]
    D = d0 + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + dn
    return D

def matrixTridiag(amp,n, h):
    A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
    B = [amp] * n
    C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
    return A, B, C

def resolveTridiag(A, B, C, D):
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

def resolveSpline(amp,n,y,H):
    a,b,c = matrixTridiag(amp,n,H)
    d = CalcD(n,H,y)
    S = resolveTridiag(a, b, c, d)
    coef = [[(S[i+1]-S[i])*H[i]*H[i]/6, S[i]*H[i]*H[i]/2, (y[i+1] - y[i] - (S[i+1]+2*S[i])*H[i]*H[i]/6), y[i]] for i in range(n-1)]

    return coef


def retPoli(x,n, coef, val):
    H = CalcH(x)
    idx = min(bisect.bisect(x, val)-1, n-2)
    z = (val - x[idx]) / H[idx]
    C = coef[idx]
    return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

if __name__ == '__main__':
    inputs, amplitude = readArgs()

    x, y = readFile(inputs)
    n = len(x)
    H = CalcH(x)
    amp = amplitude

    coef = resolveSpline(amp,n,y,H)

    for i in range(0,n-1): print(f'S{i}:',coef[i])

    X = [i for i in np.arange(0, 10, 0.01)]
    Y = [retPoli(x,n,coef,y) for y in X]

    plt.plot(x, y,'o-', color='orange')
    plt.plot(X, Y,'-b')

    plt.legend(['Linear','Cubic Spline'])
    plt.show()