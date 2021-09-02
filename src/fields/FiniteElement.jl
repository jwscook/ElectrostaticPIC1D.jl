
struct FEMGrid{BC<:AbstractBC, S<:AbstractShape, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  bases::Vector{BasisFunction{S, T}}
  partitionunityweights::Vector{T}
  function FEMGrid{BC}(N::Int, L::Float64, bases::Vector{BasisFunction{S,T}},
      puw=zeros(T, N)) where {BC<:AbstractBC, S<:AbstractShape, T}
    dummy = new{BC, S, T}(N, L, bases, puw)
    puw = solve(dummy, x->1)
    return new{BC, S, T}(N, L, bases, puw)
  end
end
function FEMGrid(N::Int, L::Float64, ::Type{S}, ::Type{BC}=PeriodicGridBC
                  ) where {BC<:AbstractBC, S<:AbstractShape}
  return FEMGrid(N, L, S(L/N), BC)
end

function FEMGrid(N::Int, L::Float64, shape::S, ::Type{BC}=PeriodicGridBC
    ) where {BC<:AbstractBC, S<:AbstractShape}
  width(shape) > L && @error ArgumentError "Shapes must not be wider than the grid"
  Δ = L / N
  bases = [BasisFunction(shape, (i-0.5) * Δ) for i ∈ 1:N]
  return FEMGrid{BC}(N, L, bases)
end
function (l::FEMGrid{BC})(x) where {BC}
  sum(i(x, BC(0.0, l.L)) * weight(i) for i ∈ l)
end
Base.size(f::FEMGrid) = (f.N,)
Base.iterate(f::FEMGrid) = iterate(f.bases)
Base.iterate(f::FEMGrid, state) = iterate(f.bases, state)
Base.getindex(l::FEMGrid, i) = l.bases[i]
bases(l::FEMGrid) = l.bases
partitionunityweights(l::FEMGrid, i) = l.partitionunityweights[i]

function Base.setindex!(l::FEMGrid, v, i)
  zero!(l.bases[i])
  l.bases[i] += v
  return l
end

function Base.isapprox(a::T, b::T, atol=0, rtol=sqrt(eps())) where {T<:FEMGrid}
  return isapprox(a.bases, b.bases, atol=atol, rtol=rtol)
end


zero!(f) = map(zero!, f)
lower(l::FEMGrid) = 0.0
upper(l::FEMGrid) = l.L
domainsize(l::FEMGrid) = lower(l) - lower(l)

#function solve(l::FEMGrid{BC}, f::F) where {BC, F}
#  p = BC(lower(l), upper(l))
#  zero!(l)
#  A = zeros(length(l), length(l))
#  for (j, v) ∈ enumerate(l.bases), (i, u) ∈ enumerate(l.bases)
#    A[i, j] = integral(u, v, p)
#  end
#  b = zeros(length(l))
#  for (j, v) ∈ enumerate(l.bases)
#    b[j] = integral(v, f, p)
#  end
#  x = A \ b
#  return x
#end
 
function solve(l::FEMGrid{BC}, f::F) where {BC, F}
  p = BC(0.0, l.L)
  A = zeros(length(l), length(l))
  ThreadsX.foreach(enumerate(l)) do (j, v)
    for (i, u) in enumerate(l)
      A[i, j] = v(centre(u),p)
    end
  end
  b = zeros(length(l))
  ThreadsX.foreach(enumerate(l)) do (j, v)
    b[j] = f(centre(v))
  end
  x = A \ b
  return x
end

function deposit!(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in lthat particle overlaps with
  bc = BC(0.0, l.L)
  qw = charge(particle) * weight(particle)
  for (index, item) ∈ intersect(basis(particle), l)
    amount = integral(item, basis(particle), bc)
    item += amount * qw * partitionunityweights(l, index)
  end
  return l
end

function antideposit(l::FEMGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in lthat particle overlaps with
  bc = BC(0.0, l.L)
  amount = 0.0
  for (index, item) ∈ intersect(basis(particle), l)
    amount += integral(item, basis(particle), bc) * partitionunityweights(l, index) * weight(item)
  end
  return amount
end


function update!(l::FEMGrid{BC}, f::F) where {BC, F}
  x = solve(l, f)
  @avx for i ∈ eachindex(x)
    setindex!(l, x[i], i)
  end
  return l
end
abstract type AbstractFEMField{BC} <: AbstractField{BC} end

struct LSFEMField{BC, S<:AbstractShape, T} <: AbstractFEMField{BC}
  charge::FEMGrid{BC,S,T}
  electricfield::FEMGrid{BC,S,T}
  function LSFEMField(charge::L, electricfield::L
      ) where {BC,S,T,L<:FEMGrid{BC,S,T}}
    @assert charge.N == electricfield.N
    @assert charge.L == electricfield.L
    return new{BC,S,T}(charge, electricfield)
  end
end
LSFEMField(a::FEMGrid) = LSFEMField(a, deepcopy(a))
Base.size(l::AbstractFEMField) = (size(l.charge),)
Base.length(l::AbstractFEMField) = length(l.charge)

lower(l::AbstractFEMField) = 0.0
upper(l::AbstractFEMField) = l.charge.L


struct GalerkinFEMField{BC, S1<:AbstractShape, S2<:AbstractShape, T
    } <: AbstractFEMField{BC}
  charge::FEMGrid{BC,S1,T}
  electricfield::FEMGrid{BC,S2,T}
  function GalerkinFEMField(charge::FEMGrid{BC,S1,T}, electricfield::FEMGrid{BC,S2,T}
      ) where {BC,S1,S2,T}
    @assert charge.L == electricfield.L
    return new{BC,S1,S2,T}(charge, electricfield)
  end
end


# Pull a special trick to solve circulant matrix that pops out of periodic BCs
"""
    gausslawsolve(f::AbstractFEMField{PeriodicGridBC})

Solve Gauss's law to obtain the electric field from the charge
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
@memoize function gausslawsolve(field::AbstractFEMField{PeriodicGridBC})
  A = massmatrix(field)
  b = forcevector(field)
  if Circulant(A) ≈ A
    c = fft(A[:, 1])
    c[1] = 1 # 1st mode (eigenvalue) is zero
    o = fft(b) ./ c
    o[1] *= false # don't divide by zero
    return real.(ifft(o))
  elseif rank(Matrix(A)) == size(A, 1) # is non-singular
    return demean!(A \ b)
  else
    u, v = eigen(Matrix(A))
    idu = 1 ./ u
    idu[end] *= false # just like ignoring 0th Fourier mode?
    A⁻¹ = real.(v * diagm(idu) * inv(v))
    return demean!(A⁻¹ * b)
  end
end
@memoize function gausslawsolve(f::AbstractFEMField{BC}) where {BC}
  A = massmatrix(f)
  b = forcevector(f)
  p = AMGPreconditioner{SmoothedAggregation}(A)
  return IterativeSolvers.cg(A, b, Pl=p)
end

postsolve!(x, ::Type{AbstractBC}) = x
postsolve!(x, ::Type{PeriodicGridBC}) = demean!(x)
function solve!(f::AbstractFEMField{BC}) where {BC}
  x = try
    gausslawsolve(f)
  catch err
    A = massmatrix(f)
    b = forcevector(f)
    @save "solve!_$(typeof(f)).jld2" f err
    @warn "Caught $err, saved field struct, and exiting."
    rethrow()
  end
  f.electricfield .= postsolve!(x, BC)
  return f
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
  return get!(()->QuadGK.quadgk(integrand,
    lower(u, v), upper(u, v), rtol=2eps())[1], cache, getkey(u, v))
end

function lsfemstiffnessmatrixintegral(u, v, cache)
  return get!(()->QuadGK.quadgk(
    x->ForwardDiff.derivative(u, x) * v(x),
    lower(u, v), upper(u, v), rtol=2eps())[1], cache, getkey(u, v))
end


function galerkinmassmatrixintegral(u::BasisFunction{GaussianShape},
                                    v::BasisFunction{GaussianShape}, cache)
  # TODO ∫ via pen & paper
  return  _galerkinmassmatrixintegral(u, v, cache)
end
function galerkinmassmatrixintegral(u::BasisFunction,
                                    v::BasisFunction, cache)
  return  _galerkinmassmatrixintegral(u, v, cache)
end
function _galerkinmassmatrixintegral(u, v, cache)
  integrand(x) = u(x) * ForwardDiff.derivative(v, x)
  return get!(()->QuadGK.quadgk(integrand,
    lower(u, v), upper(u, v), rtol=2eps())[1], cache, getkey(u, v))
end

function galerkinstiffnessmatrixintegral(u, v, cache)
  return get!(()->QuadGK.quadgk(x->u(x) * v(x),
    lower(u, v), upper(u, v), rtol=2eps())[1], cache, getkey(u, v))
end

massmatrixintegral(::LSFEMField) = lsfemmassmatrixintegral
massmatrixintegral(::GalerkinFEMField) = galerkinmassmatrixintegral
stiffnessmatrixintegral(::LSFEMField) = lsfemstiffnessmatrixintegral
stiffnessmatrixintegral(::GalerkinFEMField) = galerkinstiffnessmatrixintegral
massmatrixmatrixtype(::LSFEMField) = Symmetric
massmatrixmatrixtype(_) = SparseMatrixCSC
@memoize function massmatrix(f::AbstractFEMField)
  return matrix(f.electricfield, f.electricfield, massmatrixintegral(f),
                massmatrixmatrixtype(f))
end
@memoize function normalisedstiffnessmatrix(f::AbstractFEMField)
  return matrix(f.charge, f.charge, stiffnessmatrixintegral(f))
end

function forcevector(f::AbstractFEMField)
  return normalisedstiffnessmatrix(f) * weight.(f.charge)
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

