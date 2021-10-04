module ElectrostaticPIC1D

using Base.Threads
using ConcreteStructs
using ForwardDiff
using FFTW
using IterativeSolvers
using JLD2
using LinearAlgebra
using LoopVectorization
using Memoization
using Preconditioners
using QuadGK
using SparseArrays
using SpecialFunctions
using ThreadsX
using ToeplitzMatrices

include("BoundaryConditions.jl")

include("BasisFunctions.jl")
export BSpline, GaussianShape, TentShape, TopHatShape, DeltaFunctionShape
export BasisFunction, lower, upper, width, limits, centre, weight
include("Fields.jl")
export EquispacedValueGrid, PeriodicGridBC
export FourierField, FiniteDifferenceField, LSFEMField, GalerkinFEMField, FEMGrid
export cellcentres, solve!, update!, cell, cells, cellsize, energydensity
include("Particles.jl")
export AbstractParticle, Nuclide, Particle, pushposition!, pushvelocity!
export velocity, charge, mass, deposit!, antideposit, integral, basis, electricfield
export momentum, momentumdensity, energy, energydensity, charge, chargedensity
include("Plasmas.jl")
export Plasma, Species, positions, velocities
include("TimeIntegrators.jl")
export LeapFrogTimeIntegrator, LeapFrogTimeIntegrator, SemiImplicit2ndOrderTimeIntegrator
include("Simulations.jl")
export Simulation, SimulationParameters, run!, init!

end # module
