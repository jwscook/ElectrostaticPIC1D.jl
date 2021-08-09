module ElectrostaticPIC1D

using FFTW, LinearAlgebra, Plots, SpecialFunctions

abstract type AbstractBC end
abstract type AbstractPeriodicBC <: AsbtractBC end
struct PeriodicParticleBC <: AbstractPeriodicBC end

include("Cells.jl")
include("BasisFunctions.jl")
include("Particles.jl")
include("Fields.jl")

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
