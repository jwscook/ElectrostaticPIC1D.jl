
using ConcreteStructs, ForwardDiff, IterativeSolvers, JLD2, Lazy, LoopVectorization
using Memoization, Preconditioners, QuadGK, SparseArrays, Base.Threads
using ThreadsX, ToeplitzMatrices

abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface, the rest are in the specific files below
Base.size(g::AbstractGrid) = g.N

abstract type AbstractField{BC<:AbstractBC} end

demean!(x) = (x .-= mean(x); x)

include("fields/DeltaFunctionGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")

deposit!(f::AbstractField, p) = deposit!(f.charge, p)

function update!(f::AbstractField, species)
  for particle ∈ species
    deposit!(f.charge, particle)
  end
  return f
end
