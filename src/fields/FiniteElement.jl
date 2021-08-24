
struct LSFEMGrid{BC<:AbstractBC, S<:AbstractShape, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  bases::Vector{BasisFunction{S, T}}
  partitionunityweights::Vector{T}
  function LSFEMGrid{BC}(N::Int, L::Float64, bases::Vector{BasisFunction{S,T}},
      puw=zeros(T, N)) where {BC<:AbstractBC, S<:AbstractShape, T}
    dummy = new{BC, S, T}(N, L, bases, puw)
    puw = solve(dummy, x->1)
    return new{BC, S, T}(N, L, bases, puw)
  end
end
function LSFEMGrid(N::Int, L::Float64, ::Type{S}, ::Type{BC}=PeriodicGridBC
                  ) where {BC<:AbstractBC, S<:AbstractShape}
  return LSFEMGrid(N, L, S(L/N), BC)
end

function LSFEMGrid(N::Int, L::Float64, shape::S, ::Type{BC}=PeriodicGridBC
    ) where {BC<:AbstractBC, S<:AbstractShape}
  width(shape) > L && @error ArgumentError "Shapes must not be wider than the grid"
  Δ = L / N
  bases = [BasisFunction(shape, (i-0.5) * Δ) for i ∈ 1:N]
  return LSFEMGrid{BC}(N, L, bases)
end
function (l::LSFEMGrid{BC})(x) where {BC}
  sum(i(x, BC(0.0, l.L)) * weight(i) for i ∈ l)
end
Base.size(f::LSFEMGrid) = (f.N,)
Base.iterate(f::LSFEMGrid) = iterate(f.bases)
Base.iterate(f::LSFEMGrid, state) = iterate(f.bases, state)
Base.getindex(l::LSFEMGrid, i) = l.bases[i]

function Base.setindex!(l::LSFEMGrid, v, i)
  zero!(l.bases[i])
  l.bases[i] += v
  return l
end

zero!(f) = map(zero!, f)
lower(l::LSFEMGrid) = 0.0
upper(l::LSFEMGrid) = l.L

#function solve(l::LSFEMGrid{BC}, f::F) where {BC, F}
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
 
function solve(l::LSFEMGrid{BC}, f::F) where {BC, F}
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

function deposit!(l::LSFEMGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in lthat particle overlaps with
  bc = BC(0.0, l.L)
  qw = charge(particle) * weight(particle)
  for (index, item) ∈ intersect(basis(particle), l)
    amount = integral(item, basis(particle), bc)
    item += amount * qw * l.partitionunityweights[index]
  end
  return l
end

function update!(l::LSFEMGrid{BC}, f::F) where {BC, F}
  x = solve(l, f)
  @avx for i ∈ eachindex(x)
    setindex!(l, x[i], i)
  end
  return l
end

struct LSFEMField{BC, S<:AbstractShape, T} <: AbstractField{BC}
  charge::LSFEMGrid{BC,S,T}
  electricfield::LSFEMGrid{BC,S,T}
  function LSFEMField(charge::L, electricfield::L
      ) where {BC,S,T,L<:LSFEMGrid{BC,S,T}}
    @assert charge.N == electricfield.N
    @assert charge.L == electricfield.L
    return new{BC,S,T}(charge, electricfield)
  end
end
LSFEMField(a::LSFEMGrid) = LSFEMField(a, deepcopy(a))
Base.size(l::LSFEMField) = (size(l.charge),)
Base.length(l::LSFEMField) = length(l.charge)

lower(l::LSFEMField) = 0.0
upper(l::LSFEMField) = l.charge.L

# Pull a special trick to solve circulant matrix that pops out of periodic BCs
"""
    gausslawsolve(f::LSFEMField{PeriodicGridBC})

Solve Gauss's law to obtain the electric field from the charge
The periodic boundary condition creates a circulant Symmetric matrix,
which can be solved via FFTs (n log n).

Other solution methods are eigen decomposition (n^3) and regularisation.

...
# Arguments
- `field::LSFEMField{PeriodicGridBC}`: 
...

# Example
```julia
```
"""
@memoize function gausslawsolve(field::LSFEMField{PeriodicGridBC})
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
@memoize function gausslawsolve(f::LSFEMField{BC}) where {BC}
  A = massmatrix(f)
  b = forcevector(f)
  p = AMGPreconditioner{SmoothedAggregation}(A)
  return IterativeSolvers.cg(A, b, Pl=p)
end

postsolve!(x, ::Type{AbstractBC}) = x
postsolve!(x, ::Type{PeriodicGridBC}) = demean!(x)
function solve!(f::LSFEMField{BC, S}) where {BC, S}
  x = try
    gausslawsolve(f)
  catch err
    A = massmatrix(f)
    b = forcevector(f)
    @save "solve!_$(S)_$(f.charge.N).jld2" f err
    @warn "Caught $err, saved field struct, and exiting."
    rethrow()
  end
  f.electricfield .= postsolve!(x, BC)
  return f
end

function massmatrixintegral(u::BasisFunction{GaussianShape}, v::BasisFunction{GaussianShape}, cache)
  sigma(shape(u)) == sigma(shape(v)) || return _massmatrixintegral(u, v, cache)
  Δ = abs(centre(v) - centre(u))
  σ = sigma(shape(u))
  return exp(-(Δ/σ)^2 / 2) * (1 - (Δ/σ)^2) / sqrt(2pi) / σ^3
end
massmatrixintegral(u, v, cache) = _massmatrixintegral(u, v, cache)

function _massmatrixintegral(u, v, cache, zerohigherroranswers=false)
  integrand(x) = ForwardDiff.derivative(u, x) * ForwardDiff.derivative(v, x)
  function internal()
    (i, e) = QuadGK.quadgk(integrand, lower(u, v), upper(u, v), rtol=2eps())
    # If the relative error is too bad, then the result should probably be zero
    if zerohigherroranswers
      return i * (e < sqrt(eps()) * abs(i))
    else
      return i
    end
  end
  return get!(internal, cache, getkey(u, v))
end

@memoize function massmatrix(f::LSFEMField)
  return matrix(f.electricfield, f.electricfield, massmatrixintegral, Symmetric)
end
@memoize function normalisedstiffnessmatrix(f::LSFEMField)
  return matrix(f.charge, f.charge, stiffnessmatrixintegral)
end
function forcevector(f::LSFEMField)
  return normalisedstiffnessmatrix(f) * weight.(f.charge)
end
function stiffnessmatrixintegral(u, v, cache)
  return get!(()->QuadGK.quadgk(
    x->ForwardDiff.derivative(u, x) * v(x),
    lower(u, v), upper(u, v), rtol=2eps())[1], cache, getkey(u, v))
end

function matrix(a::LSFEMGrid{BC}, b::LSFEMGrid{BC}, integral::F, ::Type{M}=SparseMatrixCSC
               ) where {BC<:PeriodicGridBC, F, M<:AbstractMatrix}
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

