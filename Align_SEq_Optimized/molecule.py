import numpy as np

class molecule:
    '''
    Needs molecule string and temperature in [K]
    '''
    def __init__(self, mol_str, Temp):
        self.mol_str = mol_str
        self.Temp = Temp

    def params(self):
        '''
        output units
        B,D in Joules
        d_alpha in CGS
        T in rotational constants

        Output order: B, D, d_alpha, Jmax, T
        '''
        ##rotational constant in ground state in wavenumbers
        B_dict = {'O2': 1.4297, 'N2': 1.99, 'N2O': 0.4190,
                    'CO2': 0.3902, 'D2': 30.442, 'OCS': .2039}
        ##Centrifugal Distortion in ground state in wavenumbers
        D_dict = {'O2': 4.839e-6, 'N2': 5.7e-6, 'N2O': 0.176e-6,
                    'CO2': 0.135e-6, 'D2': 0, 'OCS': 0}
        ##anisotropy of the polarizability in m^3 - but cgs units
        d_alpha_dict = {'O2': 1.1e-30, 'N2': 0.93e-30, 'N2O': 2.8e-30,
                        'CO2': 2.109e-30, 'D2': .282e-30, 'OCS': 4.1e-30}
        ## Jmax for various molecules and laser < 10**13 W cm**-2
        Jmax_dict = {'O2': 20, 'N2': 20, 'N2O': 60,
                    'CO2': 60, 'D2': 35, 'OCS': 70}
        return B_dict[self.mol_str]*1.98648e-23, D_dict[self.mol_str]*1.98648e-23,\
         d_alpha_dict[self.mol_str], Jmax_dict[self.mol_str],\
         1.381e-23*self.Temp/(B_dict[self.mol_str]*1.98648e-23)

    def Boltzmann(self):
        weven_dict = {'O2': 0, 'N2': 2, 'N2O': 1, 'CO2': 1, 'D2': 2, 'OCS': 1}
        wodd_dict =  {'O2': 1, 'N2': 1, 'N2O': 1, 'CO2': 0, 'D2': 1, 'OCS': 1}

        weven = weven_dict[self.mol_str]
        wodd = wodd_dict[self.mol_str]
        pop = np.zeros(int(self.params()[3]))
        T = self.params()[4]
        for J in range(int(self.params()[3])):
            if np.mod(J,2) == 0:
                pop[J] = weven*(2*J+1)*np.exp(-J*(J+1)/T)
            else:
                pop[J] = wodd*(2*J+1)*np.exp(-J*(J+1)/T)

        return pop/np.sum(pop)
