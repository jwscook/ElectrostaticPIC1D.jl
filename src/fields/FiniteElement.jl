using ForwardDiff, Memoization, QuadGK, SparseArrays

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
  for (i, u) in enumerate(l), (j, v) in enumerate(l)
    A[i, j] = v(centre(u),p)
  end
  b = zeros(length(l))
  for (j, v) in enumerate(l)
    b[j] = f(centre(v))
  end
  x = A \ b
  for i ∈ eachindex(x)
    @assert weight(l.bases[i]) == 0
    l.bases[i] += x[i]
  end
  return l, A, b, x
end


struct LSFEMField{BC<:PeriodicGridBC, S<:AbstractShape, T} <: AbstractField{BC}
  charge::LSFEMGrid{BC,S,T}
  electricfield::LSFEMGrid{BC,S,T}
end
LSFEMField(a::LSFEMGrid) = LSFEMField(a, deepcopy(a))
Base.size(l::LSFEMField) = (size(l.charge),)
Base.length(l::LSFEMField) = l.charge.N

lower(l::LSFEMField) = 0.0
upper(l::LSFEMField) = (@assert l.charge.L == l.electricfield.L; l.charge.L)

@memoize union(f::LSFEMField, b::BasisFunction) = filter(i->overlap(i, b), f.bases)

function deposit!(f::LSFEMField, particle::AbstractParticle)
  for bases ∈ union(f, BasisFunction(particle))
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

function solve!(f::LSFEMField)
  @memoize gausslawsolve(f) = massmatrix(f) \
    Matrix(normalisedstiffnessmatrix(f)) # shouldn't need to convert to dense
  f.electricfield .= (gausslawsolve(f) * f.charge)
end

@memoize function massmatrix(f::LSFEMField{PeriodicGridBC})
  p = PeriodicBCHandler(lower(f), upper(f))
  M = spzeros(length(f), length(f))
  for (j, v) ∈ enumerate(f.electricfield), (i, u) ∈ enumerate(f.electricfield)
    M[i, j] = QuadGK.quadgk(x̄->(x = p(x̄); 
      ForwardDiff.derivative(u, x) * ForwardDiff.derivative(v, x)),
      lower(u, v), upper(u, v))[1]
  end
  return M
end
weights(f) = [weight(i) for i in f]
function stiffnessmatrix(f::LSFEMField)
  return normalisedstiffnessmatrix(f) * weights(f)
end

@memoize function normalisedstiffnessmatrix(f::LSFEMField)
  p = PeriodicBCHandler(lower(f), upper(f))
  K = spzeros(length(f), length(f))
  for (j, v) ∈ enumerate(f.charge), (i, u) ∈ enumerate(f.electricfield)
    K[i, j] = QuadGK.quadgk(x̄->(x = p(x̄);
      ForwardDiff.derivative(u, x) * v(x)),
      lower(u, v), upper(u, v))[1]
  end
  return K
end

