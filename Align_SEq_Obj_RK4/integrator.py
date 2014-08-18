import numpy as np
from scipy.integrate import ode
from numpy.linalg import norm

class integrator():
    '''
    Needs Jmax, sigma, Delta_omega, B, D
    '''
    def __init__(self, Jmax, sigma, strength, B, D):
        self.Jmax = Jmax
        self.sigma = sigma
        self.strength = strength
        self.B = B
        self.D = D

    def c2(self,J,M):
        '''
        Matrix Element
        J -> J
        '''
        if abs(M)>J:
            return 0
        else:
            return 1.0/3 + 2.0/3*((J*(J+1)-3*M**2))/((2*J+3)*(2*J-1))

    def cp2(self, J,M):
        '''
        Matrix Element
        J -> J+2
        '''
        if abs(M)>J:
            return 0
        else:
            return ((2.0*J+1)*(2*J+5)*(J+1-M))**.5*((2+J-M)*(J+1+M)*(J+2+M))**.5/((2*J+1)*(2*J+3)*(2*J+5))

    def cm2(self, J,M):
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

    def rhs(self,t, x, Mparm):
        '''
        RHS of schrodinger equation.
        '''
        dx = np.array(np.zeros(self.Jmax, dtype = 'complex'))
        if (t >= 0  and  t <= 2*self.sigma):
            Delta_omega = self.strength*np.sin(np.pi*t/2/self.sigma)**2
        else:
            Delta_omega = 0

        for k in range(self.Jmax):
            if k == 0 or k == 1:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k]*self.c2(k,Mparm) -
                    Delta_omega*x[k+2]*self.cp2(k,Mparm))
            elif k == self.Jmax - 2 or k == self.Jmax-1:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k-2]*self.cm2(k,Mparm) -
                    Delta_omega*x[k]*self.c2(k,Mparm))
            else:
                dx[k] = -1j*(x[k]*(k*(k+1) - self.D/self.B*k**2*(k+1)**2 - Delta_omega) -
                    Delta_omega*x[k-2]*self.cm2(k,Mparm) -
                    Delta_omega*x[k+2]*self.cp2(k,Mparm) -
                    Delta_omega*x[k]*self.c2(k,Mparm))
        return dx

    def advance(self, t, dt, x, M):
        k1 = self.rhs(t,x,M)
        k2 = self.rhs(t+.5*dt, x+.5*dt*k1,M)
        k3 = self.rhs(t+.5*dt, x+.5*dt*k2,M)
        k4 = self.rhs(t+dt, x + dt*k3,M)
        x += dt/6*(k1+2*k2+2*k3+k4)
        return x


    def integrate(self):
        '''
        Integrate Schrodinger Equation with SciPy ODE solver

        returns an array of wavefunction coefficients at the end of the integration period.
        '''
        ## Initialize
        tend = 2*self.sigma; dt = .04*self.sigma;t = 0
        tt = np.linspace(0,5,1000)
        cos2 = np.zeros(tt.size,dtype = 'complex')
        Cstor = np.zeros((self.Jmax,int(2*self.Jmax+1), self.Jmax), dtype = 'complex')
        start = np.zeros(self.Jmax, dtype = 'complex')

        ## Integrate Schrodinger Eq. Loop over all initial wavefunctions |J,M>
        for J in range(self.Jmax):
            for M in range(J+1):
                #initialize
                t = 0
                init = 0*start
                init[J] = 1
                psi = self.advance(0, dt, init, M)
                #integrate
                while t < tend:
                    t+=dt
                    psi = self.advance(t,dt,psi, M)

                #store
                Cstor[J,M,:] = psi/norm(psi)**.5
                Cstor[J,-M,:] = psi/norm(psi)**.5
        return Cstor
