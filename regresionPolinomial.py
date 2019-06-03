import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as S

file = "Examen_Segundo_Parcial_ICCSO/exam_A_dataset.csv"
data = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[0,1])
X = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[0])
Y = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[1])



def polyfit2(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    A=np.hstack([c1,c2,c3])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

#print(polyfit2(X,Y,2))

def polyfit3(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4=np.power(c2,3)
    A=np.hstack([c1,c2,c3,c4])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

def polyfit4(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4=np.power(c2,3)
    c5=np.power(c2,4)
    A=np.hstack([c1,c2,c3,c4,c5])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit4(X,Y,2))

x = S.symbols('x')
y = S.symbols('y')


y = -3.26599160e+00 + 5.53685798e-01 * x -1.96329350e+00 * pow(x,2) + 5.21537693e-01*pow(x,3) + -2.39372653e-03* pow(x,4)

f= S.lambdify(x,y,'math')
yen= f(X)
print (yen)
yout=yen.astype(list)

plt.scatter(X,Y,color = 'blue')
plt.scatter(X,yout,color = 'yellow')
plt.show()