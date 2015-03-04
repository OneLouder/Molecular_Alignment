from ffa_sim import *
import matplotlib.pyplot as plt
from pylab import *
# import pyximport; pyximport.install()
plt.close('all')
rh = 'off'
simulation = ffa_sim('N2O', 70, 120e-15, .1, rh)
t, cos2, theta,tt, rho = simulation.run_sim()

if rh == 'off':
    plt.show()
    
elif rh == 'on':
    # f, ax = plt.subplots(1,2, figsize = (20,8))
    # ax[1].imshow(np.real(rho[:,::-1].transpose()), extent = [min(theta), max(theta*180/np.pi),  min(tt), max(tt)], 
    #           aspect='auto', interpolation = 'bilinear')
    # ax[0].plot(-1*cos2, t, 'k-')
    # ax[0].set_ylim([0, max(t)])
    # ax[1].grid()
    # plt.grid()
    # plt.show()

    f, ax = plt.subplots(2,1, figsize = (10,8))
    ax[1].imshow(np.real(rho), extent = [  min(tt), max(tt), max(theta*180/np.pi),min(theta)], 
              aspect='auto', interpolation = 'bilinear')
    ax[0].plot(t, cos2, 'k-')
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim([0,max(t)])
    ax[1].grid()
    plt.grid()
    plt.show()
else:
    pass
