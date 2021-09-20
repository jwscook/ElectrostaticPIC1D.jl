
using ConcreteStructs, ForwardDiff, IterativeSolvers, JLD2, Lazy, LoopVectorization
using Memoization, Preconditioners, QuadGK, SparseArrays, Base.Threads
using ThreadsX, ToeplitzMatrices

abstract type AbstractGrid{BC,T} <: AbstractVector{T} end

# AbstractArray interface, the rest are in the specific files below
Base.size(g::AbstractGrid) = g.N

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

function energydensity(f::AbstractField{BC}; rtol=sqrt(eps())) where {BC}
  bc = BC(0.0, domainsize(f))
  return quadgk(x->f.electricfield(x)^2, 0, domainsize(f), rtol=rtol)[1] / 2 / domainsize(f) # TODO - make faster
end
function chargedensity(f::AbstractField{BC}) where {BC}
  bc = BC(domainsize(f))
  return mapreduce(b->integral(b, x->1, bc) * weight(b), +, bases(f.chargedensity)) / domainsize(f)
end

function Base.isapprox(a::T, b::T, atol=0, rtol=sqrt(eps())) where {T<:AbstractField}
  return isapprox(a.chargedensity, b.chargedensity, atol=atol, rtol=rtol) &&
    isapprox(a.electricfield, b.electricfield, atol=atol, rtol=rtol)
end

zerochargedensity!(f::AbstractField) = zero!(f.chargedensity)
zeroelectricfield!(f::AbstractField) = zero!(f.electricfield)

function update!(f::AbstractField, species)
  for particle ∈ species
    deposit!(f.chargedensity, particle)
  end
  return f
end

function cellcentres(f::AbstractField)
  return ((1:numberofunknowns(f)) .- 0.5) * domainsize(f) / numberofunknowns(f)
end

function chargedensityvalues(f::AbstractField, x=cellcentres(f))
  return map(f.chargedensity, x)
end
function electricfieldvalues(f::AbstractField, x=cellcentres(f))
  return map(f.electricfield, x)
end
