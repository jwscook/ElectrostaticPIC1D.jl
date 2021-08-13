struct Nuclide
  charge::Float64
  mass::Float64
  numberdensity::Float64
end

abstract type AbstractParticle end

mutable struct Particle{S<:AbstractShape, BC<:AbstractBC} <: AbstractParticle
  nuclide::Nuclide
  x::Float64
  v::Float64
  weight::Float64
  shape::S
  bc::BC
end

weight(p::Particle) = p.weight
shape(p::Particle) = p.shape

function BasisFunction(p::Particle{S}) where {S<:AbstractShape}
  return BasisFunction{S}(shape(p), p.x, 1.0)
end


