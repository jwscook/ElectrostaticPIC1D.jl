module ElectrostaticPIC1D

using FFTW, LinearAlgebra, SpecialFunctions

abstract type AbstractBC end
abstract type AbstractPeriodicBC <: AbstractBC end
struct PeriodicParticleBC <: AbstractPeriodicBC end

include("Cells.jl")
include("BasisFunctions.jl")
include("Particles.jl")
include("Fields.jl")
export DeltaFunctionGrid, PeriodicGridBC, FourierField
export cellcentres, solve!

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
