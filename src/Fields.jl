using ConcreteStructs, Memoization
abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface, the rest are in the specific files below
Base.size(g::AbstractGrid) = g.N

abstract type AbstractField{BC<:AbstractBC} end

demean!(x) = (x .-= mean(x); x)

include("fields/DeltaFunctionGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")
