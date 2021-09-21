
struct Species{S<:AbstractShape}
  particles::Vector{Particle{S}}
end

@concrete struct Plasma
  species
end

for (type, field) ∈ ((:Plasma, :species), (:Species, :particles))
  for fun ∈ (:size, :iterate, :getindex, :length)
    @eval Base.$(fun)(x::$(type)) = $(fun)(x.$(field))
  end
  for fun ∈ (:iterate, :getindex, :lastindex)
    @eval Base.$(fun)(x::$(type), y) = $(fun)(x.$(field), y)
  end
end

positions(s::Species) = [position(p) for p ∈ s]
positions(p::Plasma) = [positions(s) for s ∈ p]
velocities(s::Species) = [velocity(p) for p ∈ s]
velocities(p::Plasma) = [velocities(s) for s ∈ p]
weights(s::Species) = [weight(p) for p ∈ s]
weights(p::Plasma) = [weights(s) for s ∈ p]

energy(s::Species) = sum(weight(p) * velocity(p)^2 * mass(p)/2 for p ∈ s)
momentum(s::Species) = sum(weight(p) * velocity(p) * mass(p) for p ∈ s)
charge(s::Species) = sum(weight(p) * charge(p) for p ∈ s)

energy(p::Plasma) = sum(energy.(p))
momentum(p::Plasma) = sum(momentum.(p))
charge(p::Plasma) = sum(charge.(p))

function maxspeed(p::Plasma)
  output = 0.0
  for species ∈ p.species, particle ∈ species
    output = max(output, abs(velocity(particle)))
  end
  return output
end
function Base.copy!(a::Plasma, b::Plasma)
  for (i, species) ∈ enumerate(b), (j, particle) ∈ enumerate(b)
    copy!(a[i][j], particle)
  end
  return a
end

function deposit!(field::AbstractField, plasma::Plasma)
  for species ∈ plasma, particle ∈ species
    deposit!(field, particle)
  end
  return field
end

for op ∈ (:push!, :pushvelocity!)
  @eval function $(op)(plasma::Plasma, field::AbstractField, dt)
    for species ∈ plasma, particle ∈ species
      $(op)(particle, field, dt)
    end
    return plasma
  end
end

for op ∈ (:pushposition!,)
  @eval function $(op)(plasma::Plasma, dt, bc)
    for species ∈ plasma, particle ∈ species
      $(op)(particle, dt, bc)
    end
    return plasma
  end
end

