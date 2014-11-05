Molecular_Alignment
===================
A poorly organized overview of various implementations of a simulation to study field free molecular alignment in simple molecuels. I kept all the different version organized this way to show my logical progression from a pure python implementation to one implementing dramatic speed ups using Cython. This is more so I don't forget what I did so I can re-learn from it in the future. This "project" was more of a training excercise that ultimately lead to simulations I needed in my research. Use and learn from at your own risk. 

## Align_SEq_Optimized
The optimized version is the most complete.  It includes a Cython version of the molecular alignment code with the C files already constructed.  It doesn't require pyximport.



### The other versions
#### Align_SEq 
A simple pure python implementation.  Uses a Scipy integrator to solve the ODEs.
#### Align_SEq
A pure python implementation that uses a home-built RK4 solver.
#### Align_SEq_Obj
A simple pure python version that uses classes to better organize the code. Uses Scipy to integrate ODEs.
#### Align_SEq_Obj_RK4
A pure python implementation that uses classes to organize the code and an RK4 solver to integrate the ODEs.
#### Align_SEq_Obj_RK4_Cython
A Cython implementation of the code that speeds up the ODE solver dramatically.  This version requires pyximport to run since the C code doesn't come pre-compiled.  Has the same speed as the Optimized version.  If tweaking the code is necessary, it will be easy to use this version since the pyzimport takes care of all the rebuilding and recompiling of the C code every time the program is run.  
