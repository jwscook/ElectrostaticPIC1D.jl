using ConcreteStructs, Memoization
abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface
Base.size(g::AbstractGrid) = g.N

# Periodic BC
struct PeriodicGridBC <: AbstractPeriodicBC end

abstract type AbstractField{BC<:AbstractBC} end

demean!(x) = (x .-= mean(x); x)

include("fields/DeltaFunctionGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")
