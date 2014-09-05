Molecular_Alignment
===================

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
