import numpy as np
cimport numpy as np
cimport cython
from scipy.special import sph_harm
from integrator import *
from laser import *
# from Wigner3J import *
import matplotlib.pyplot as plt
from const import *

class expectation_values(integrator):
    
    '''
    Evaluation of thermally averaged, time-dependent expectation values.
    '''
    def cos2(self, np.ndarray[np.complex64_t, ndim = 3] Cstor, 
        np.ndarray[np.float_t, ndim = 1, negative_indices = False] Jweight):
        # Type defs
        cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] cos2
        cdef np.ndarray[np.float_t, ndim = 1] tt
        cdef int J, M, jj
        cdef float w, phi
        # Init
        tt = np.linspace(0,5,1000)
        cos2 = np.zeros(tt.size,dtype = 'complex64')
        for J in range(self.Jmax):
            for M in range(-J,J+1):
                for jj in range(self.Jmax-2):
                    w = 4*jj+6
                    phi = np.angle(Cstor[J,M,jj])-np.angle(Cstor[J,M,jj+2])
                    cos2 += Jweight[J]/(2*J+1)*(abs(Cstor[J,M,jj])**2*self.c2(jj,M) +
                            abs(Cstor[J,M,jj])*abs(Cstor[J,M,jj+2])*self.cp2(jj,M)*np.cos(w*tt+phi))
        return tt, cos2

    def cos2plot(self, tt, cos2, molecule):
        plt.figure()
        plt.plot(tt*hbar/self.B*10**12,np.real(cos2),'k-')
        plt.xlabel('Time [ps]')
        plt.ylabel(r'$\langle$ cos$^2 \theta \rangle$')
        plt.ylim(0,1)
        # plt.show() 

    def rho(self, np.ndarray[np.complex64_t, ndim = 3] Cstor, 
        np.ndarray[np.float_t, ndim = 1, negative_indices = False] Jweight):
        
        # Type defs
        cdef np.ndarray[np.float_t, ndim = 1, negative_indices = False] js, jj1
        # cdef np.ndarray[np.complex64_t, ndim = 1, negative_indices = False] jj1
        cdef np.ndarray[np.complex64_t, ndim = 2, negative_indices = False] rh
        cdef np.ndarray[np.complex64_t, ndim = 5, negative_indices = False] temp
        cdef np.ndarray[np.complex64_t, ndim = 4, negative_indices = False] temp2, temp3
        cdef np.ndarray[np.complex64_t, ndim = 3, negative_indices = False] temp4, 
        cdef np.ndarray[np.float_t, ndim = 1, negative_indices = False] tt, theta, phi
        cdef float w, dtheta, dphi
        cdef int J, M, th, ph
        
        # Init
        tt = np.linspace(0,5,100)
        theta = np.linspace(0, np.pi / 2 , 10)
        dtheta = theta[1]-theta[0]
        phi = np.linspace(0, np.pi, 5)
        dphi = phi[1] - phi[0]

        temp = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1, len(phi), len(tt) ),dtype = 'complex64')
        temp2 = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1,len(tt) ),dtype = 'complex64')
        temp3 = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1, len(tt) ),dtype = 'complex64')
        temp4 = np.zeros((len(theta), self.Jmax, len(tt) ),dtype = 'complex64')
        rh = np.zeros((len(theta), len(tt) ),dtype = 'complex64')
        
        # Ugliest loop ever.
        for J in range(self.Jmax):
            for M in range(0,J+1):  
                js = np.arange(float(abs(M)), float(self.Jmax))
                jj1 = js * (js + 1)
                for th in range(len(theta)):
                    for ph in range(len(phi)):
                        #add up jjs and phis
                        temp[th,J,M,ph,:] =  np.sum(np.array( np.multiply(np.mat((-1j) ** M *\
                            Cstor[J,M,abs(M):self.Jmax] * 2 * sph_harm(M, js, phi[ph], theta[th])).T,\
                            np.exp( np.mat(jj1).T *  np.mat(-1j*tt)) )),axis = 0)
                #integrate phis
                    temp2[th,J,M,:] = np.sum(np.multiply(temp[th,J,M,:,:],\
                            np.conjugate(temp[th,J,M,:,:])), axis = 0) * 2 * dphi
                #mag square of all jjs, M's, and weight
                    temp3[th, J, M, :] = (temp2[th, J, M, :]  * Jweight[J] /\
                            ( 2 * J + 1) ) * np.sin(theta[th])
        #sum them up.
        temp4[:,:,:] = np.sum(temp3[:, :,:, :], axis = 2) 
        rh[:,:]  = np.sum(temp4[:, :, :], axis = 1) 

        return tt, theta, rh

    def rho_old(self, np.ndarray[np.complex64_t, ndim = 3] Cstor, 
        np.ndarray[np.float_t, ndim = 1, negative_indices = False] Jweight):
        
        # Type defs
        cdef np.ndarray[np.complex64_t, ndim = 2, negative_indices = False] rh
        cdef np.ndarray[np.complex64_t, ndim = 5, negative_indices = False] temp
        cdef np.ndarray[np.complex64_t, ndim = 4, negative_indices = False] temp2, temp3
        cdef np.ndarray[np.complex64_t, ndim = 3, negative_indices = False] temp4
        cdef np.ndarray[np.float_t, ndim = 1, negative_indices = False] tt, theta, phi, js, jj1
        cdef float w, dtheta, dphi
        cdef int J, M, jj, th
        
        # Init
        tt = np.linspace(0,5,100)
        theta = np.linspace(0, np.pi / 2 , 10)
        dtheta = theta[1]-theta[0]
        phi = np.linspace(0, np.pi * 2, 9)
        dphi = phi[1] - phi[0]

        temp = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1, len(phi), len(tt) ),dtype = 'complex64')
        temp2 = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1,len(tt) ),dtype = 'complex64')
        temp3 = np.zeros((len(theta), self.Jmax, self.Jmax * 2 +1, len(tt) ),dtype = 'complex64')
        temp4 = np.zeros((len(theta), self.Jmax, len(tt) ),dtype = 'complex64')
        rh = np.zeros((len(theta), len(tt) ),dtype = 'complex64')
        
        # Ugliest loop ever.
        for J in range(self.Jmax):
            for M in range(-J,J+1):  
                js = np.arange(float(abs(M)), float(self.Jmax+1))
                jj1 = js * (js + 1)
                for th in range(len(theta)):
                    for ph in range(len(phi)):
                        for jj in range(abs(M),self.Jmax):
                            #add up jjs and phis
                            temp[th,J,M,ph,:] += Cstor[J,M,jj] * np.exp( -1j * jj * (jj + 1) * tt ) * (-1j) ** M * sph_harm(M, jj, phi[ph], theta[th])
                #integrate phis
                    temp2[th,J,M,:] = np.sum(np.multiply(temp[th,J,M,:,:],np.conjugate(temp[th,J,M,:,:])), axis = 0) * dphi
                #mag square of all jjs, M's, and weight
                    temp3[th, J, M, :] = (temp2[th, J, M, :]  * Jweight[J] / ( 2 * J + 1) )*1#   np.sin(theta[th])
        #sum them up.
        temp4[:,:,:] = np.sum(temp3[:, :,:, :], axis = 2) 
        rh[:,:]  = np.sum(temp4[:, :, :], axis = 1) 

        return tt, theta, rh



        
