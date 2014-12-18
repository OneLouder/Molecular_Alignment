##Wigner 3 J symbol using Racah Formula
#Craig Benko 2014.04.30
import numpy as np
cimport numpy as np
cimport cython

def Wigner3J(int j1, int j2, int j3, int m1, int m2, int m3):
    cdef float wig
    cdef int t, t1, t2, t3, t4, t5, tmin, tmax

    if 2*j1 != np.floor(2*j1) or 2*j2 != np.floor(2*j2) or 2*j3 != np.floor(2*j3) or 2*m1 != np.floor(2*m1) or 2*m2 != np.floor(2*m2) or 2*m3 != np.floor(2*m3):
        # print('All arguments must be integers')
        wig = 0
    elif j1 - m1 != np.floor(j1 -m1):
        # print('2*j1 must have same parity as 2*m1')
        wig = 0
    elif j2 - m2 != np.floor(j2 -m2):
        # print('2*j2 must have same parity as 2*m2')
        wig = 0
    elif j3 - m3 != np.floor(j3 -m3):
        # print('2*j3 must have same parity as 2*m3')
        wig = 0
    elif j3 > j1 + j2 or j3 < np.abs(j1 - j2):
        # print('j3 out of bounds')
        wig = 0
    elif np.abs(m1) > j1:
        # print('m1 is out of bounds')
        wig = 0
    elif np.abs(m2) > j2:
        # print('m2 is out of bounds')
        wig = 0
    elif np.abs(m3) > j3:
        # print('m3 is out of bounds')
        wig = 0
    else:
        t1 = j2 - m1 - j3
        t2 = j1 + m2 - j3
        t3 = j1 + j2 - j3
        t4 = j1 - m1
        t5 = j2 + m2

        tmin = max(0, max(t1, t2))
        tmax = min(t3, min(t4, t5))

        wig = 0

        for t in range(tmin, tmax+1):
            wig += (-1)**t / (np.math.factorial(t)*np.math.factorial(t-t1)*\
                np.math.factorial(t-t2)*np.math.factorial(t3-t)*np.math.factorial(t4-t)*\
                np.math.factorial(t5-t) )

        wig *= (-1)**(j1-j2-m3)*np.sqrt(np.math.factorial(j1+j2-j3)*\
            np.math.factorial(j1-j2+j3)*np.math.factorial(-j1+j2+j3)/\
            np.math.factorial(j1+j2+j3+1)*np.math.factorial(j1+m1)*\
            np.math.factorial(j1-m1)*np.math.factorial(j2+m2)*np.math.factorial(j2-m2)*\
            np.math.factorial(j3+m3)*np.math.factorial(j3-m3))

    return wig
