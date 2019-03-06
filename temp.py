import numpy as np


def func(x, A):
    print(A)
    return x



if __name__ == '__main__':
    trans = np.vectorize(func, excluded=['A'])
    a = np.array([[1,2], [3,4]])
    X = np.array([[5,6],[7,8]])
    trans(a, A=X)

    map(lambda x: print(x), a)