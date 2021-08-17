# ElectrostaticPIC1D.jl

*WIP: This is a nascent package and subject to large chages. See tests for current capabilities*


## Purpose

Test the interaction of different field solvers, time integrators and particle
shapes to aid in the understanding of more complicated related methods and
in particular conservation properties.

Attention is restricted at first to the 1D periodic electrostatic case.

The plan is implement:

 - Field Solvers
   - [x] Finite difference
   - [x] Least square finite element
   - [ ] mixed finite element
   - [ ] Fourier (but not fully-Spectral)

 - Time integrators
   - [ ] Verlet (leap-frog)
   - [ ] Fixed point Crank-Nicolson-esque semi-implicit
   - [ ] Fixed point 3rd order Simpson-esque semi-implicit

 - Particle shapes
   - [ ] Delta function
   - [ ] BSpline 0
   - [ ] BSpline 1
   - [ ] BSpline 2
   - [ ] Gaussian
