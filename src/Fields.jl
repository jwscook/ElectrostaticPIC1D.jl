
using ConcreteStructs, ForwardDiff, IterativeSolvers, JLD2, Lazy, LoopVectorization
using Memoization, Preconditioners, QuadGK, SparseArrays, Base.Threads
using ThreadsX, ToeplitzMatrices

abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface, the rest are in the specific files below
#Base.size(g::AbstractGrid) = g.N

abstract type AbstractField{BC<:AbstractBC} end

demean!(x) = (x .-= mean(x); x)

include("fields/EquispacedValueGrids.jl")
include("fields/Fourier.jl")
include("fields/FiniteDifference.jl")
include("fields/FiniteElement.jl")

deposit!(f::AbstractField, p) = deposit!(f.chargedensity, p)
antideposit(f::AbstractField, p) = antideposit(f.electricfield, p)
electricfield(f::AbstractField, p) = antideposit(f.electricfield, p)
chargedensity(f::AbstractField, p) = antideposit(f.chargedensity, p)
cellsize(l::AbstractField) = l.chargedensity.L / numberofunknowns(l.chargedensity)
domainsize(l::AbstractField) = l.chargedensity.L
numberofunknowns(f::AbstractField) = numberofunknowns(f.chargedensity)

function energy(f::AbstractField{BC}; rtol=100eps()) where {BC}
  bc = BC(0.0, domainsize(f))
  return quadgk(x->f.electricfield(x)^2, 0, domainsize(f), rtol=rtol)[1] / 2
end
function charge(f::AbstractField{BC}) where {BC}
  bc = BC(domainsize(f))
  return mapreduce(b->integral(b, x->1, bc) * weight(b), +, bases(f.chargedensity))
end
energydensity(f::AbstractField; rtol=100eps()) = energy(f; rtol=rtol) / domainsize(f)
chargedensity(f::AbstractField) = charge(f) / domainsize(f)

function cellcentres(f::AbstractField)
  return ((1:numberofunknowns(f)) .- 0.5) * domainsize(f) / numberofunknowns(f)
end
