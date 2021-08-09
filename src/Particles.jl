struct Nuclide
  charge::Float64
  mass::Float64
  numberdensity::Float64
end

mutable struct Particle{S<:AbstractShape, BC<:AbstractBC}
  nuclide::Nuclide
  x::Float64
  v::Float64
  weight::Float64
  shape::S
  bc::BC
end

weight(p::Particle) = p.weight
shape(p::Particle) = p.shape
support(p::Particle{DeltaFunction}) = (p.x, p.x)
support(p::Particle{TopHat}) = p.x .+ (-1/2, 1/2) .* p.shape.fullwidth
support(p::Particle{GaussianShape}) = p.x .+ (-6, 6) .* p.shape.σ


function charge(p::Particle, limits)
  return integral(BasisFunction(tophat(limits[2] - limits[2]), mean(limits)),
                  BasisFunction(particle))
end

function charge(b::BasisFunction, p::Particle, _=nothing)
  return integral(b, BasisFunction(particle))
end

