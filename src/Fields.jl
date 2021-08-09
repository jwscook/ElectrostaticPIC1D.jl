using ConcreteStructs
abstract type AbstractGrid{BC} <: AbstractArray end

# AbstractArray interface
Base.size(g::AbstractGrid) = g.N

# Periodic BC
struct PeriodicGridBC <: AsbtractPeriodicBC end

include("fields/DiracDeltaGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")
