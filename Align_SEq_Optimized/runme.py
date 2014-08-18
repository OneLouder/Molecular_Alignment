from ffa_sim import *
# import pyximport; pyximport.install()

simulation = ffa_sim('N2', 70, 130e-15, .1)
simulation.run_sim()
