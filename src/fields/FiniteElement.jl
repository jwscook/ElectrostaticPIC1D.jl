using ForwardDiff, JLD2, Memoization, QuadGK, SparseArrays, Base.Threads, ThreadsX

struct LSFEMGrid{BC<:AbstractBC, S<:AbstractShape, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  bases::Vector{BasisFunction{S, T}}
end
function LSFEMGrid(N::Int, L::Float64, ::Type{S}, ::Type{BC}=PeriodicGridBC
                  ) where {BC<:AbstractBC, S<:AbstractShape}
  return LSFEMGrid(N, L, S(L/N), BC)
end
function LSFEMGrid(N::Int, L::Float64, shape::S, ::Type{BC}=PeriodicGridBC
    ) where {BC<:AbstractBC, S<:AbstractShape}
  S <: GaussianShape && @assert N >= 12
  Δ = L / N
  bases = [BasisFunction(shape, (i-0.5) * Δ) for i ∈ 1:N]
  return LSFEMGrid{BC,S,Float64}(N, L, bases)
end
function (l::LSFEMGrid{PeriodicGridBC})(x)
  sum(i(x, PeriodicBCHandler(0.0, l.L)) * weight(i) for i ∈ l)
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

#function update!(l::LSFEMGrid, f::F) where {F}
#  p = PeriodicBCHandler(lower(l), upper(l))
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
#  for i ∈ eachindex(x)
#    @assert weight(l.bases[i]) == 0
#    l.bases[i] += x[i]
#  end
#  return l
#end
function update!(l::LSFEMGrid, f::F) where {F}
  p = PeriodicBCHandler(0.0, l.L)
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
  for i ∈ eachindex(x)
    @assert weight(l.bases[i]) == 0
    l.bases[i] += x[i]
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
Base.length(l::LSFEMField) = l.charge.N

lower(l::LSFEMField) = 0.0
upper(l::LSFEMField) = l.charge.L

function deposit!(f::LSFEMField, particle::AbstractParticle)
  for bases ∈ (BasisFunction(particle) ∈ f)
    for basis ∈ bases
      basis += integral(BasisFunction(particle), basis) * particle.charge *
        particle.weight
    end
  end
end

function update!(f::LSFEMField, species)
  for particle ∈ species
    deposit!(f, particle)
  end
end

@memoize gausslawsolve(f) = massmatrix(f) \
  Matrix(normalisedstiffnessmatrix(f)) # shouldn't need to convert to dense

postsolve!(x, ::Type{AbstractBC}) = x
postsolve!(x, ::Type{PeriodicGridBC}) = demean!(x)
function solve!(f::LSFEMField{BC, S}) where {BC, S}
  A = try
    gausslawsolve(f)
  catch err
    M = massmatrix(f)
    K = Matrix(normalisedstiffnessmatrix(f))
    @save "solve!_$(S)_$(f.charge.N).jld2" f err
    @warn "Caugh $err, saved matrices, and exiting."
    rethrow()
  end
  f.electricfield .= postsolve!(A * f.charge, BC)
  return f
end

function massmatrixintegrand(u, v, cache)
  return get!(()->QuadGK.quadgk(
    x->ForwardDiff.derivative(u, x) * ForwardDiff.derivative(v, x),
    lower(u, v), upper(u, v), rtol=10eps())[1], cache, getkey(u, v))
end
@memoize function massmatrix(f::LSFEMField)
  return matrix(f.electricfield, f.electricfield, massmatrixintegrand)
end
@memoize function normalisedstiffnessmatrix(f::LSFEMField)
  return matrix(f.charge, f.charge, stiffnessmatrixintegrand)
end
function stiffnessmatrix(f::LSFEMField)
  return normalisedstiffnessmatrix(f) * weight.(f.charge)
end
function stiffnessmatrixintegrand(u, v, cache)
  return get!(()->QuadGK.quadgk(
    x->ForwardDiff.derivative(u, x) * v(x),
    lower(u, v), upper(u, v), rtol=10eps())[1], cache, getkey(u, v))
end

function matrix(a::LSFEMGrid{BC}, b::LSFEMGrid{BC}, integrand::F
               ) where {BC<:PeriodicGridBC, F}
  p = PeriodicBCHandler(lower(a), upper(a))
  A = spzeros(length(a), length(a))
  caches = [IdDict{UInt64,Any}() for _ ∈ 1:Threads.nthreads()]
  foreach(enumerate(b)) do (j, v̄)
    cache = caches[Threads.threadid()]
    for (i, ū) ∈ enumerate(a)
      u, v = translate(ū, v̄, p)
#      (j-1 <= i <= j+ 1) && @show i, j, u, v
      u ∈ v || continue
      A[i, j] = integrand(u, v, cache)
    end
  end
  return A
end

