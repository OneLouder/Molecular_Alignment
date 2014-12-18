import numpy as np
cimport numpy as np
cimport cython
from scipy.integrate import ode
from numpy.linalg import norm
from scipy.special import sph_harm

class integrator():
    '''
    Needs Jmax, sigma, Delta_omega, B, D
    '''
    def __init__(self, int Jmax, float sigma, float strength, float B, float  D):
        self.Jmax = Jmax
        self.sigma = sigma
        self.strength = strength
        self.B = B
        self.D = D

    def c2(self, float J, float M):
        '''
        Matrix Element
        J -> J
        '''
        if abs(M)>J:
            return 0
        else:
            return 1.0/3 + 2.0/3*((J*(J+1)-3*M**2))/((2*J+3)*(2*J-1))

    def cp2(self, float J, float M):
        '''
        Matrix Element
        J -> J+2
        '''
        if abs(M)>J:
            return 0
        else:
            return ((2.0*J+1)*(2*J+5)*(J+1-M))**.5*((2+J-M)*(J+1+M)*(J+2+M))**.5/((2*J+1)*(2*J+3)*(2*J+5))

    def cm2(self, float J, float M):
        '''
        Matrix Element
        J -> J-2
        '''
        if abs(M)>J:
            return 0
        elif M==J:
            return 0
        else:
            return ((2.0*J-3)*(2*J+1)*(J-1-M))**.5*((J-M)*(J-1+M)*(J+M))**.5/((2*J-3)*(2*J-1)*(2*J+1))

    def rhs(self, float t, np.ndarray[np.complex64_t, ndim = 1] x, float Mparm):
        '''
        RHS of schrodinger equation.
        '''
        cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] dx
        cdef float Delta_omega

        dx = np.array(np.zeros(self.Jmax, dtype = 'complex64'))
        if (t >= 0  and  t <= 2*self.sigma):
            Delta_omega = self.strength*np.sin(np.pi*t/2/self.sigma)**2
        else:
            Delta_omega = 0.0

        cdef int k
        for k in range(self.Jmax):
            if k == 0 or k == 1:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k]*self.c2(k,Mparm) - Delta_omega*x[k+2]*self.cp2(k,Mparm))
            elif k == self.Jmax - 2 or k == self.Jmax-1:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k-2]*self.cm2(k,Mparm) - Delta_omega*x[k]*self.c2(k,Mparm))
            else:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k-2]*self.cm2(k,Mparm) - Delta_omega*x[k+2]*self.cp2(k,Mparm) -
                    Delta_omega*x[k]*self.c2(k,Mparm))
        return dx

    def advance(self, float t,float dt, np.ndarray[np.complex64_t, ndim = 1] x, float M):
        cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] k1, k2, k3, k4
        k1 = self.rhs(t,x,M)
        k2 = self.rhs(t+.5*dt, x+.5*dt*k1,M)
        k3 = self.rhs(t+.5*dt, x+.5*dt*k2,M)
        k4 = self.rhs(t+dt, x + dt*k3,M)
        x += dt/6*(k1+2*k2+2*k3+k4)
        return x


    def integrate(self):
        '''
        Integrate Schrodinger Equation with ODE solver

        returns an array of wavefunction coefficients at the end of the integration period.
        '''
        ## Initialize
        cdef float tend, dt, t
        cdef int JJ, MM
        cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] cos2, start, init
        cdef np.ndarray[np.complex64_t, ndim = 3] Cstor 
        cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] psi
        cdef np.ndarray[np.float_t, ndim = 1] tt
        ## define
        tend = 2*self.sigma; dt = .04*self.sigma;t = 0
        tt = np.linspace(0,5,1000)
        cos2 = np.zeros(tt.size, dtype = 'complex64')
        Cstor = np.zeros((self.Jmax, int(2*self.Jmax+1), self.Jmax), dtype = 'complex64')
        start = np.zeros(self.Jmax, dtype = 'complex64')

        ## Integrate Schrodinger Eq. Loop over all initial wavefunctions |J,M>
        for JJ in range(self.Jmax):
            for MM in range(JJ+1):
                #initialize
                t = 0
                init = 0*start
                init[JJ] = 1.0
                psi = self.advance(0, dt, init, MM)
                #integrate
                while t < tend:
                    t+=dt
                    psi = self.advance(t,dt,psi, MM)

                #store
                Cstor[JJ,MM,:] = psi/norm(psi)**.5
                Cstor[JJ,-MM,:] = psi/norm(psi)**.5

        return Cstor
