using Statistics
struct FEMGrid{BC<:AbstractBC, S<:AbstractShape, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  bases::Vector{BasisFunction{S, T}}
  partitionunityweights::Vector{T}
  buckets::Dict{BasisFunction{TopHatShape,Float64},Set{Int}}
  function FEMGrid{BC}(N::Int, L::Float64, bases::Vector{BasisFunction{S,T}},
      puw=zeros(T, N)) where {BC<:AbstractBC, S<:AbstractShape, T}
    bc = BC(L)
    puw = 1 ./ [sum(v(centre(u), bc) for u ∈ bases) for v ∈ bases]
    @assert !any(iszero, puw)
    @assert all(isfinite, puw)

    zero!.(bases)

    # Split the grid up into buckets;
    # Any basisfunction that does not intersect with a key
    # will also not intersect with any element in the associated value vector.
    # Therefore to find all bases on the grid intersecting with a basis, B,
    # simply find the key(s), k, that B intersects with and loop through the
    # associated value, which itselfis a vector of grid bases interesecting
    # with the key, k.
    # N.B. TopHatShape is used simply as an interval
    nbins = Int(round(sqrt(N))) # make the buckets approx √N "cells" wide
    bins = collect(0:nbins) ./ nbins * L
    buckets = Dict{BasisFunction{TopHatShape,Float64}, Set{Int}}()
    bc = PeriodicGridBC(L)
    for i ∈ 2:length(bins)
      l, u = bins[i-1], bins[i]
      bin = BasisFunction{TopHatShape,Float64}(TopHatShape(u - l), (l + u)/2, NaN) # weight is unimportant
      contents = Set{Int}()
      for (index, item) in enumerate(bases)
        in(translate(item, bin, bc)...) && push!(contents, index)
      end
      buckets[bin] = contents
    end

    return new{BC, S, T}(N, L, bases, puw, buckets)
  end
end

function FEMGrid(N::Int, L::Float64, shape::S, ::Type{BC}=PeriodicGridBC
    ) where {BC<:AbstractBC, S<:AbstractShape}
  width(shape) > L && @error ArgumentError "Shapes must not be wider than the grid"
  Δ = L / N
  bases = [BasisFunction(shape, (i-0.5) * Δ) for i ∈ 1:N]
  return FEMGrid{BC}(N, L, bases)
end

function (l::FEMGrid{BC})(x) where {BC}
  # This is equivalent to return sum(i(x, BC(0.0, l.L)) * weight(i) for i ∈ l)
  # but is faster
  bc = BC(l.L)
  processedindices = Set{Int}()
  amount = 0.0
  for (k, v) ∈ l.buckets
    in(x, k) || continue
    for index ∈ filter(x->!(x ∈ processedindices), v)
      item = l.bases[index]
      amount += item(x, bc) * weight(item)
      push!(processedindices, index)
    end
  end
  return amount
end
Base.size(f::FEMGrid) = (f.N,)
Base.iterate(f::FEMGrid) = iterate(f.bases)
Base.iterate(f::FEMGrid, state) = iterate(f.bases, state)
Base.getindex(l::FEMGrid, i) = l.bases[i]
bases(l::FEMGrid) = l.bases

function Base.setindex!(l::FEMGrid, v, i::Integer)
  zero!(l.bases[i])
  l.bases[i] += v
  return l
end

lower(l::FEMGrid) = 0.0
upper(l::FEMGrid) = l.L
zero!(f::FEMGrid) = map(zero!, f)
domainsize(l::FEMGrid) = l.L
numberofunknowns(l::FEMGrid) = length(l.bases)

#function solve(l::FEMGrid{BC}, f::F) where {BC, F}
#  bc = BC(lower(l), upper(l))
#  zero!(l)
#  A = zeros(length(l), length(l))
#  for (j, v) ∈ enumerate(l.bases), (i, u) ∈ enumerate(l.bases)
#    A[i, j] = integral(u, v, bc)
#  end
#  b = zeros(length(l))
#  for (j, v) ∈ enumerate(l.bases)
#    b[j] = integral(v, f, bc)
#  end
#  x = A \ b
#  return x
#end

# Pull a special trick to solve circulant matrix that pops out of periodic BCs
function solve(A::AbstractMatrix, b::AbstractVector)
  x = if Circulant(A) ≈ A
    c = fft(A[:, 1])
    c[1] = 1 # 1st mode (eigenvalue) is zero; will * by false later
    o = fft(b) ./ c
    o[1] *= false # PeriodicGridBC means we want zero DC offset
    for i ∈ eachindex(o) # incase we are dividing by any other zeros
      isnan(o[i]) && (o[i] *= false)
    end
    real.(ifft(o))
  elseif rank(Matrix(A)) == size(A, 1) # is non-singular
    A \ b
  else
    u, v = eigen(Matrix(A))
    idu = 1 ./ u
    idu[end] *= false # just like ignoring 0th Fourier mode?
    A⁻¹ = real.(v * diagm(idu) * inv(v))
    A⁻¹ * b
  end
  @assert all(isfinite, x)
  return x
end

"""
    fastdeposit!(l::FEMGrid{BC},particle)where{BC<:AbstractBC}

description

...
# Arguments
- `l::FEMGrid{BC}`:
- `particlewhere{BC<:AbstractBC}`:
...

# Equivalent to
```julia
function deposit!(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  bc = BC(l.L)
  qw = charge(particle) * weight(particle)
  for (index, item) ∈ enumerate(l)
    item += integral(item, basis(particle), bc) * qw * 
      l.partitionunityweights[index]
  end
  return l
end
```
"""
function fastdeposit!(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  bc = BC(l.L)
  qw = charge(particle) * weight(particle)
  # Only deposit from particle into each basis function once
  # Record which basis functions have recevied deposition in this set
  processedindices = Set{Int}()
  for (k, v) ∈ (l.buckets)
    overlap(k, basis(particle), bc) || continue
    for index ∈ filter(x->!(x ∈ processedindices), v)
      item = l.bases[index]
      item += integral(item, basis(particle), bc) * qw *
        l.partitionunityweights[index]
      push!(processedindices, index)
    end
  end
  return l
end


"""
    fastantideposit(l::FEMGrid{BC},particle)where{BC<:AbstractBC}

description

...
# Arguments
- `l::FEMGrid{BC}`: 
- `particlewhere{BC<:AbstractBC}`: 
...

# Example
```julia
function antideposit(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  bc = BC(l.L)
  amount = 0.0
  for (index, item) ∈ enumerate(l)
    amount += integral(item, basis(particle), bc) * weight(item)
  end
  return amount
end
```
"""
function fastantideposit(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  bc = BC(l.L)
  # Only antideposit from each basis function once
  # Record which basis functions have been processed already
  processedindices = Set{Int}()
  amount = 0.0
  for (k, v) ∈ l.buckets
    overlap(k, basis(particle), bc) || continue
    for index ∈ filter(x->!(x ∈ processedindices), v)
      item = l.bases[index]
      amount += integral(item, basis(particle), bc) * weight(item)
      push!(processedindices, index)
    end
  end
  return amount
end


function deposit!(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  return fastdeposit!(l, particle)
  #bc = BC(l.L)
  #qw = charge(particle) * weight(particle)
  #for (index, item) ∈ enumerate(l)
  #  item += integral(item, basis(particle), bc) * qw *
  #    l.partitionunityweights[index]
  #end
  #return l
end

function antideposit(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  return fastantideposit(l, particle)
  #bc = BC(l.L)
  #amount = 0.0
  #for (index, item) ∈ enumerate(l)
  #  amount += integral(item, basis(particle), bc) * weight(item)
  #end
  #return amount
end


function update!(l::FEMGrid{BC}, f::F) where {BC, F}
  bc = BC(0.0, l.L)

  A = zeros(length(l), length(l))
  ThreadsX.foreach(enumerate(l)) do (j, v)
    for (i, u) in enumerate(l)
      A[i, j] = v(centre(u), bc)
    end
  end

  b = zeros(length(l))
  ThreadsX.foreach(enumerate(l)) do (j, v)
    b[j] = f(centre(v))
  end

  x = A \ b

  @avx for i ∈ eachindex(x)
    setindex!(l, x[i], i)
  end

  return l
end

abstract type AbstractFEMField{BC} <: AbstractField{BC} end

"""
    GalerkinFEMField

Using basis function, u to represent both the charge density the
the electric field, solve for the latter from the former via:
    ∫ u' v'dx = ∫ u' v dx.
"""
struct LSFEMField{BC, S<:AbstractShape, T} <: AbstractFEMField{BC}
  chargedensity::FEMGrid{BC,S,T}
  electricfield::FEMGrid{BC,S,T}
  function LSFEMField(chargedensity::L, electricfield::L
      ) where {BC,S,T,L<:FEMGrid{BC,S,T}}
    @assert chargedensity.N == electricfield.N
    @assert chargedensity.L == electricfield.L
    return new{BC,S,T}(chargedensity, electricfield)
  end
end
LSFEMField(a::FEMGrid) = LSFEMField(a, deepcopy(a))
LSFEMField(N::Int, L::Real, shape::S) where {S<:AbstractShape} = LSFEMField(FEMGrid(N, L, shape))

numberofunknowns(l::AbstractFEMField) = numberofunknowns(l.chargedensity)

zero!(f::AbstractFEMField) = map(zero!, (f.chargedensity, f.electricfield))

"""
    GalerkinFEMField

Using basis function, u, to represent the charge density and v to represent the
the electric field, solve for the latter from the former via:
    -∫ u' v'dx = ∫ u v' dx.
"""
struct GalerkinFEMField{BC, S1<:AbstractShape, S2<:AbstractShape, T
    } <: AbstractFEMField{BC}
  chargedensity::FEMGrid{BC,S1,T}
  electricfield::FEMGrid{BC,S2,T}
  function GalerkinFEMField(chargedensity::FEMGrid{BC,S1,T}, electricfield::FEMGrid{BC,S2,T}
      ) where {BC,S1,S2,T}
    @assert chargedensity.L == electricfield.L
    return new{BC,S1,S2,T}(chargedensity, electricfield)
  end
end
function GalerkinFEMField(N::Int, L::Real, chargeshape::S1, efieldshape::S2
  ) where {S1<:AbstractShape, S2<:AbstractShape}
  return GalerkinFEMField(FEMGrid(N, L, chargeshape), FEMGrid(N,L, efieldshape))
end

function solve(field::AbstractFEMField{PeriodicGridBC})
  A = massmatrix(field)
  b = forcevector(field)
  return solve(A, b)
end

function solve(f::AbstractFEMField{BC}) where {BC}
  A = massmatrix(f)
  b = forcevector(f)
  p = AMGPreconditioner{SmoothedAggregation}(A)
  return IterativeSolvers.gmres(A, b, Pl=p)
end

postsolve!(x, ::Type{AbstractBC}) = x
postsolve!(x, ::Type{PeriodicGridBC}) = demean!(x)

"""
    solve!(f::AbstractFEMField{PeriodicGridBC})

Solve for the electric field from the chargedensity.
The periodic boundary condition creates a circulant Symmetric matrix,
which can be solved via FFTs (n log n).

Other solution methods are eigen decomposition (n^3) and regularisation.

...
# Arguments
- `field::AbstractFEMField{PeriodicGridBC}`: 
...

# Example
```julia
```
"""
function solve!(f::AbstractFEMField{BC}) where {BC}
  x = try
    solve(f)
  catch err
    A = massmatrix(f)
    b = forcevector(f)
    @save "solve!_$(typeof(f)).jld2" f err
    @warn "Caught $err, saved field struct, and exiting."
    rethrow(err)
  end
  f.electricfield .= postsolve!(x, BC)
  return f
end

function quadrature(integrand::F, intervals) where {F}
  return QuadGK.quadgk(integrand, intervals..., order=7, rtol=2eps())[1]
end

function lsfemmassmatrixintegral(u::BasisFunction{GaussianShape},
                                 v::BasisFunction{GaussianShape}, cache)
  if !(sigma(shape(u)) == sigma(shape(v)))
    return _lsfemmassmatrixintegral(u, v, cache)
  end
  Δ = abs(centre(v) - centre(u))
  σ = sigma(shape(u))
  return exp(-(Δ/σ)^2 / 2) * (1 - (Δ/σ)^2) / sqrt(2pi) / σ^3
end
function lsfemmassmatrixintegral(u::BasisFunction, v::BasisFunction, cache)
  return _lsfemmassmatrixintegral(u, v, cache)
end
function _lsfemmassmatrixintegral(u, v, cache) 
  integrand(x) = ForwardDiff.derivative(u, x) * ForwardDiff.derivative(v, x)
  return get!(()->quadrature(integrand, knots(u, v)), cache, getkey(u, v))
end

function lsfemstiffnessmatrixintegral(u, v, cache)
  integrand(x) = ForwardDiff.derivative(u, x) * v(x)
  return get!(()->quadrature(integrand, knots(u, v)), cache, getkey(u, v))
end

function galerkinmassmatrixintegral(u::BasisFunction,
                                    v::BasisFunction, cache)
  return  _galerkinmassmatrixintegral(u, v, cache)
end
function _galerkinmassmatrixintegral(u, v, cache)
  integrand(x) = -ForwardDiff.derivative(u, x) * ForwardDiff.derivative(v, x)
  return get!(()->quadrature(integrand, knots(u, v)), cache, getkey(u, v))
end

function galerkinstiffnessmatrixintegral(u, v, cache)
  integrand(x) = u(x) * ForwardDiff.derivative(v, x)
  return get!(()->quadrature(integrand, knots(u, v)), cache, getkey(u, v))
end

massmatrixintegral(::LSFEMField) = lsfemmassmatrixintegral
massmatrixintegral(::GalerkinFEMField) = galerkinmassmatrixintegral
stiffnessmatrixintegral(::LSFEMField) = lsfemstiffnessmatrixintegral
stiffnessmatrixintegral(::GalerkinFEMField) = galerkinstiffnessmatrixintegral
massmatrixmatrixtype(::LSFEMField) = Symmetric
massmatrixmatrixtype(_) = SparseMatrixCSC
@memoize function massmatrix(f::AbstractFEMField)
  return matrix(f.chargedensity, f.electricfield, massmatrixintegral(f),
                massmatrixmatrixtype(f))
end
@memoize function normalisedstiffnessmatrix(f::AbstractFEMField)
  return matrix(f.chargedensity, f.chargedensity, stiffnessmatrixintegral(f))
end

function forcevector(f::AbstractFEMField)
  return normalisedstiffnessmatrix(f) * weight.(f.chargedensity)
end


function matrix(a::FEMGrid{BC}, b::FEMGrid{BC}, integral::F,
    ::Type{M}=SparseMatrixCSC) where {BC<:PeriodicGridBC, F, M<:AbstractMatrix}
  bc = BC(lower(a), upper(a))
  As = [spzeros(length(a), length(a)) for _ ∈ 1:Threads.nthreads()]
  caches = [IdDict{UInt64,Any}() for _ ∈ 1:Threads.nthreads()]
  foreach(enumerate(b)) do (j, v̄)
    cache = caches[Threads.threadid()]
    A = As[Threads.threadid()]
    for (i, ū) ∈ enumerate(a)
      M == Symmetric && j > i && (A[i, j] = A[j, i]; continue)
      u, v = translate(ū, v̄, bc)
      u ∈ v || continue
      A[i, j] = integral(u, v, cache)
    end
  end
  return M(sum(As))
end

