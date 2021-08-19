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
export DeltaFunctionGrid, PeriodicGridBC
export FourierField, FiniteDifferenceField, LSFEMField, LSFEMGrid
export cellcentres, solve!, update!
include("Particles.jl")
export AbstractParticle, Nuclide, Particle, pushposition!, pushvelocity!
export velocity, charge, mass, deposit!, integral, basis

struct Species{S<:AbstractShape}
  particles::Vector{Particle{S}}
end

struct Simulation{F<:AbstractField}
  species::Vector{Species}
  timestep::Float64
  endtime::Float64
  diagnoticeverytimestep::Int64
  field::F
end

end # module
