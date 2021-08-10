using ConcreteStructs
abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface
Base.size(g::AbstractGrid) = g.N

# Periodic BC
struct PeriodicGridBC <: AbstractPeriodicBC end

abstract type AbstractField{BC<:AbstractBC} end

include("fields/DeltaFunctionGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")
