from ffa_sim import *
# import pyximport; pyximport.install()

simulation = ffa_sim('N2O', 250, 130e-15, .02)
t, cos = simulation.run_sim()

