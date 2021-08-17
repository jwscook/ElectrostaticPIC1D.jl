struct Nuclide
  charge::Float64
  mass::Float64
end

charge(n::Nuclide) = charge(n.charge)
mass(n::Nuclide) = mass(n.mass)

abstract type AbstractParticle end

mutable struct Particle{S<:AbstractShape} <: AbstractParticle
  nuclide::Nuclide
  basis::BasisFunction{S}
  x::Float64
  v::Float64
end
function Particle(n::Nuclide, b::BasisFunction{S}=DeltaFunctionShape) where {S}
  return Particle(n, b, 0.0, 0.0)
end

weight(p::Particle) = p.basis.weight
shape(p::Particle) = p.basis.shape
charge(p::Particle) = charge(p.nuclide)
mass(p::Particle) = mass(p.nuclide)

function BasisFunction(p::Particle{S}) where {S<:AbstractShape}
  return BasisFunction{S}(shape(p), p.x, 1.0)
end


