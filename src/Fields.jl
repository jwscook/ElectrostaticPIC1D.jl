
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

deposit!(f::AbstractField, p) = deposit!(f.charge, p)
electricfield(f::AbstractField, p) = antideposit(f.electricfield, p)
antideposit(f::AbstractField, p) = antideposit(f.electricfield, p)
cellsize(l::AbstractField) = l.charge.L / numberofunknowns(l.charge)
domainsize(l::AbstractField) = l.charge.L 
numberofunknowns(f::AbstractField) = numberofunknowns(f.charge)

function energydensity(f::AbstractField{BC}; rtol=sqrt(eps())) where {BC}
  bc = BC(0.0, domainsize(f))
  return quadgk(x->f.electricfield(x)^2, 0, domainsize(f), rtol=rtol)[1] / 2 / domainsize(f) # TODO - make faster
end
function chargedensity(f::AbstractField{BC}) where {BC}
  bc = BC(0.0, domainsize(f))
  #@time a = mapreduce(i->integral(i, x->1, bc) * weight(i), +, bases(f.charge)) / domainsize(f)
  #@time b = quadgk(x->f.charge(x), 0, domainsize(f), rtol=eps())[1] / domainsize(f) # TODO - make faster
  #@show b / a
  return mapreduce(b->integral(b, x->1, bc) * weight(b), +, bases(f.charge)) / domainsize(f)
end

function Base.isapprox(a::T, b::T, atol=0, rtol=sqrt(eps())) where {T<:AbstractField}
  return isapprox(a.charge, b.charge, atol=atol, rtol=rtol) && 
    isapprox(a.electricfield, b.electricfield, atol=atol, rtol=rtol)
end

zerocharge!(f::AbstractField) = zero!(f.charge)
zeroelectricfield!(f::AbstractField) = zero!(f.electricfield)

function update!(f::AbstractField, species)
  for particle ∈ species
    deposit!(f.charge, particle)
  end
  return f
end

#function Base.intersect(x::BasisFunction,
#                        g::AbstractGrid{BC}) where {BC}
#  bc = BC(0.0, domainsize(g))
#  accept(b) = (ab = translate(x, b, bc); in(ab...))
#  return ((i, b) for (i, b) ∈ enumerate(g) if accept(b))
#end
#
function cellcentres(f::AbstractField)
  return ((1:numberofunknowns(f)) .- 0.5) * domainsize(f) / numberofunknowns(f)
end

function chargevalues(f::AbstractField, x=cellcentres(f))
  return map(f.charge, x)
end
function electricfieldvalues(f::AbstractField, x=cellcentres(f))
  return map(f.electricfield, x)
end
