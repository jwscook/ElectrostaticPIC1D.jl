using Memoization, QuadGK

struct LSFEMGrid{BC<:AbstractBC, S<:AbstractShape, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  bc::BC
  bases::Vector{BasisFunction{S, T}}
end
Base.size(f::LSFEMGrid) = f.N
Base.iterate(f::LSFEMGrid) = iterate(f.bases)
Base.iterate(f::LSFEMGrid, state) = iterate(f.bases, state)

function LSFEMGrid(N::Int, L::Float64, shape::AbstractShape, ::Type{BC}=PeriodicGridBC) where {BC<:AbstractBC}
  Δ = L / N
  bases = [BasisFunction(deepcopy(shape), (i-0.5) * Δ) for i ∈ 1:n]
  return LSFEMGrid(N, L, BC(N,L), bases)
end

struct LSFEMField{BC<:PeriodicGridBC, T, S<:AbstractShape} <: AbstractField{BC}
  charge::LSFEMGrid{BC,T}
  electricfield::LSFEMGrid{BC,T}
end

@memoize union(f::LSFEMField, b::BasisFunction) = filter(i->overlap(i, b), f.bases)

function deposit!(f::LSFEMField, particle)
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

@memoize function massmatrix(f::LSFEMField)
  M = zeros(size(f), size(f))
  for (v, j) ∈ enumerate(f), (u, i) ∈ enumerate(f)
    M[i, j] = QuadGK.quadgk(x-> ForwardDiff.gradient(u, x) * 
                                ForwardDiff.gradient(v, x),
                                lower(u, v), upper(u, v))[1]
  end
  return M
end
weights(f) = [weight(i) for i in f]
function stiffnessmatrix(f::LSFEMField)
  return normalisedstiffnessmatrix(f) * weights(f)
end

@memoize function normalisedstiffnessmatrix(f::LSFEMField)
  K = zeros(size(f), size(f))
  for (v, j) ∈ enumerate(f), (u, i) ∈ enumerate(f)
    K[i, j] = QuadGK.quadgk(x-> ForwardDiff.gradient(u, x) * v(x),
                            lower(u, v), upper(u, v))[1]
  end
  return K
end

