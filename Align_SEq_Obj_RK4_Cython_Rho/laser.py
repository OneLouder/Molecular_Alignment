import numpy as np

class laser:
    '''
    Takes pulse_FWHM in [s] and Int in [10**14 W cm**-2]

    pulse(self, t, strength, sigma) outputs a sin**2 pulse for the normalized strength and sigma.

    '''
    def __init__(self, pulse_FWHM , Int ):
        self.pulse_FWHM = pulse_FWHM
        self.Int = Int

    def pulse(self, t, strength, sigma):
        if (t >= 0  and  t <= 2*sigma):
            p = strength*np.sin(np.pi*t/2/sigma)**2
        else:
            p = 0
        return p
