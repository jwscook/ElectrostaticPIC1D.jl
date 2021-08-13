module ElectrostaticPIC1D

using FFTW, LinearAlgebra, SpecialFunctions

include("BoundaryConditions.jl")

include("BasisFunctions.jl")
export BSpline, GaussianShape, TentShape, TopHatShape, DeltaFunctionShape
include("Particles.jl")
include("Fields.jl")
export DeltaFunctionGrid, PeriodicGridBC
export FourierField, FiniteDifferenceField, LSFEMField, LSFEMGrid
export cellcentres, solve!, update!

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
