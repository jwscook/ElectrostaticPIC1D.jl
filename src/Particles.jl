struct Nuclide
  charge::Float64
  mass::Float64
end

abstract type AbstractParticle end

mutable struct Particle{S<:AbstractShape} <: AbstractParticle
  nuclide::Nuclide
  x::Float64
  v::Float64
  basis::BasisFunction{S}
end

weight(p::Particle) = p.basis.weight
shape(p::Particle) = p.basis.shape

function BasisFunction(p::Particle{S}) where {S<:AbstractShape}
  return BasisFunction{S}(shape(p), p.x, 1.0)
end


