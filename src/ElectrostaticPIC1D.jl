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
export FourierField, FiniteDifferenceField, LSFEMField, LSFEMGrid
export cellcentres, solve!, update!, cell, cells
include("Particles.jl")
export AbstractParticle, Nuclide, Particle, pushposition!, pushvelocity!
export velocity, charge, mass, deposit!, integral, basis, electricfield
include("Plasmas.jl")
include("TimeIntegrators.jl")
include("Simulations.jl")

end # module
