
struct Species{S<:AbstractShape}
  particles::Vector{Particle{S}}
end
Base.size(s::Species) = size(s.particles)
Base.iterate(s::Species) = iterate(s.particles)
Base.iterate(s::Species, state) = iterate(s.particles, state)

@concrete struct Plasma
  species
end
Base.size(s::Plasma) = size(s.species)
Base.iterate(s::Plasma) = iterate(s.species)
Base.iterate(s::Plasma, state) = iterate(s.species, state)


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
  @eval function $(op)(plasma::Plasma, dt)
    for species ∈ plasma, particle ∈ species
      $(op)(particle, dt)
    end
    return plasma
  end
end

