import numpy as np
import time

# import my classes

# cimport numpy as np
from laser import *
from molecule import *
from integrator import *
from const import *
from expectation_values import *

# close old plots.
# plt.close('all')


class ffa_sim:

    '''
    simulator of field free molecular alignment.  Needs molecule [str], temperature [K], pulse width [s] and intensity[10**14 W cm**-2].
    Default values are provided so you can simply run_sim after creating an instance of the ffa_sim class.
    '''

    def __init__(self, molecule='N2', temperature=100, pulse=100e-15, intensity=.1):
        self.molecule = molecule
        self.temperature = temperature
        self.pulse = pulse
        self.intensity = intensity

    def run_sim(self):
        # start time
        timer = time.time()
        # create molecule
        mol = molecule(self.molecule, self.temperature)
        # create laser
        las = laser(self.pulse, self.intensity)

        # generate usefule quantities
        Jmax = mol.params()[3]
        delta_alpha = mol.params()[2]
        B = mol.params()[0]
        D = mol.params()[1]
        Jweight = mol.Boltzmann()

        sigma = las.pulse_FWHM * B / hbar
        E0 = 2.74 * 10 ** 10 * las.Int ** .5  # electric field amplitude
        strength = 0.5 * 4 * np.pi * epsilon0 * delta_alpha * E0 ** 2 / B

        # create integrator
        intSEq = integrator(
            mol.params()[3], las.pulse_FWHM * B / hbar, strength, B, D)

        # integrate
        Cstor = intSEq.integrate()

        # Expectation value, incoherent, thermal average.
        exp_val = expectation_values(Jmax, sigma, strength, B, D)
        tt, cos2 = exp_val.cos2(Cstor, Jweight)
        # end timer
        elapsed = time.time() - timer
        print('\n Program took ' + str(round(elapsed)) + ' s to run. \n')
        # plot
        exp_val.cos2plot(tt, cos2, mol.mol_str)

        # End program
        return tt*hbar/B*10**12, cos2

    def save_sim(self, tt=0, cos2=0):
        '''
        Save Data
        
        filename string format:
            molecule_temperature_pulseDuration_intensity.txt

            temperature [K]
            pulseDuration [s]
            intensity [10**14 W cm**-2]

        '''
        filename = self.molecule+'_'+str(self.temperature)+'_'+str(self.pulse)+'_'+\
            str(self.intensity).replace('.','p')+'.txt'
        np.savetxt(filename, np.column_stack((tt,cos2)), delimiter = ',')
        return filename
        print('Saved File: ' + filename)
    
    def load_sim(self, filename = 'dummy.txt'):
        return np.genfromtxt(filename, delimiter = ',', dtype = 'complex64') 

