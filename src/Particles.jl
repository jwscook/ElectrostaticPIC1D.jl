
struct Nuclide
  charge::Float64
  mass::Float64
  charge_over_mass::Float64
end
Nuclide(q, m) = Nuclide(q, m, q / m)

charge(n::Nuclide) = n.charge
mass(n::Nuclide) = n.mass
charge_mass_ratio(n::Nuclide) = n.charge_over_mass

abstract type AbstractParticle end

mutable struct Particle{S<:AbstractShape} <: AbstractParticle
  nuclide::Nuclide
  basis::BasisFunction{S}
  velocity::Float64
end
function Particle(n::Nuclide, shape::AbstractShape=DeltaFunctionShape();
    kwargstuple...) where {S<:AbstractShape}
  kwargs = Dict(kwargstuple)
  haskey(kwargs, :x) && (kwargs[:position] = kwargs[:x])
  haskey(kwargs, :v) && (kwargs[:velocity] = kwargs[:v])
  haskey(kwargs, :w) && (kwargs[:weight] = kwargs[:w])
  position = get(kwargs, :position, 0.0)
  velocity = get(kwargs, :velocity, 0.0)
  weight = get(kwargs, :weight, 0.0)
  basis = BasisFunction(shape, position, weight)
  return Particle(n, basis, velocity)
end

velocity(p::Particle) = p.velocity
basis(p::Particle) = p.basis
Base.position(p::Particle) = centre(p)

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


function pushposition!(p::Particle, dt, bc::AbstractBC)
  translate!(p.basis, p.velocity * dt, bc)
  return p
end
function pushvelocity!(p::Particle, E::Number, dt)
  p.velocity += charge_mass_ratio(p) * E * dt
  return p
end
import Base.push!
Base.push!(p::Particle, E, dt, bc) = (pushposition!(p, dt, bc); pushvelocity!(p, E, dt); p)

function pushvelocity!(p::Particle, f::AbstractField, dt)
  E = electricfield(f, p)
  @assert isfinite(E)
  return pushvelocity!(p, E, dt)
end

