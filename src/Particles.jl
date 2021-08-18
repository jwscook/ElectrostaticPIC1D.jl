
struct Nuclide
  charge::Float64
  mass::Float64
  charge_over_mass::Float64
  Nuclide(q, m) = new(q, m, q / m)
end

charge(n::Nuclide) = n.charge
mass(n::Nuclide) = n.mass
charge_mass_ratio(n::Nuclide) = n.charge_over_mass

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

Base.position(p::Particle) = p.x
velocity(p::Particle) = p.v
basis(p::Particle) = p.basis
centre(p::Particle) = centre(basis(p))

for op ∈ (:weight, :lower, :upper)
  @eval $op(p::Particle) = $op(p.basis)
end
for op ∈ (:mass, :charge, :charge_mass_ratio)
  @eval $op(p::Particle) = $op(p.nuclide)
end

pushposition!(p::Particle, dt) = (p.x += p.v * dt; p)
pushvelocity!(p::Particle, E, dt) = (p.v += charge_mass_ratio(p) * E * dt; p)
Base.push!(p::Particle, E, dt) = (pushposition!(p, dt); pushvelocity!(p, E, dt); p)

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

function deposit!(obj::AbstractGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in obj that particle overlaps with
  bc = BC(0.0, obj.L)
  @show intersect(particle, obj, bc)
  for item ∈ intersect(particle, obj, bc)
    amount = integral(item, basis(particle), bc) * charge(particle) * weight(particle)
    @show item, particle, amount
    @show overlap(item, basis(particle))
    item += amount
  end
  return obj
end




