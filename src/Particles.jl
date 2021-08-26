
struct Nuclide
  charge::Float64
  mass::Float64
  charge_over_mass::Float64
end
Nuclide(q, m) = Nuclide(q, m, q / m)

charge(n::Nuclide) = n.charge
mass(n::Nuclide) = n.mass
charge_mass_ratio(n::Nuclide) = n.charge_over_mass

function Base.copy!(a::Nuclide, b::Nuclide)
  a.charge = b.charge
  a.mass = b.mass
  a.charge_over_mass = b.charge_over_mass
  return a
end
abstract type AbstractParticle end

mutable struct Particle{S<:AbstractShape} <: AbstractParticle
  nuclide::Nuclide
  basis::BasisFunction{S}
  velocity::Float64
end
function Particle(n::Nuclide, shape::AbstractShape=DeltaFunctionShape();
    position::Float64=0.0, velocity::Float64=0.0, weight::Float64=0.0
    ) where {S<:AbstractShape}
  basis = BasisFunction(shape, position, weight)
  return Particle(n, basis, velocity)
end

velocity(p::Particle) = p.velocity
basis(p::Particle) = p.basis
Base.position(p::Particle) = centre(p)

function Base.copy!(a::Particle, b::Particle)
  copy!(a.nuclide, b.nuclide)
  copy!(a.basis, b.basis)
  a.velocity = b.velocity
  return a
end

for op ∈ (:weight, :lower, :upper, :width, :centre)
  @eval $op(p::Particle) = $op(basis(p))
end
for op ∈ (:mass, :charge, :charge_mass_ratio)
  @eval $op(p::Particle) = $op(p.nuclide)
end
for op ∈ (:-, :+)
  @eval function Base.$op(a::Particle{S}, b::Particle{S}) where {S<:AbstractShape}
    @assert a.nuclude == b.nuclude
    return Particle(a.nuclude, $op(a.basis, b.basis), $op(a.velocity, b.velocity))
  end    
end

for op ∈ (:*, :/)
  @eval function Base.$op(a::Particle, x::Number)
    return Particle(a.nuclude, $op(a.basis, x), $op(a.velocity, x))
  end    
end


function pushposition!(p::Particle, dt)
  p.basis = translate(p.basis, p.velocity * dt)
  return p
end
function pushvelocity!(p::Particle, E::Number, dt)
  p.velocity += charge_mass_ratio(p) * E * dt
  return p
end
Base.push!(p::Particle, E, dt) = (pushposition!(p, dt); pushvelocity!(p, E, dt); p)

function pushvelocity!(p::Particle, f::AbstractField, dt)
  @show p, typeof(f)
  @show typeof(electricfield(f, p))
  return pushvelocity!(p, electricfield(f, p), dt)
end

BasisFunction(p::Particle) = p.basis

function Base.getindex(p::Particle, i)
  i == 1 && return lower(p)
  i == 2 && return upper(p)
  throw(BoundsError("getindex(::Particle, 1) returns the lower extent of the
                    particle's basis function, and getindex(::Particle, 2)
                    returns the upper extent. 
                    Index requested is $i."))
end

overlap(b::BasisFunction, p::Particle) = overlap(b, basis(p))
Base.intersect(p::Particle, x) = intersect(basis(p), x)


