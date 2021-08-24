
struct Species{S<:AbstractShape}
  particles::Vector{Particle{S}}
end

@concrete struct Plasma
  species
end
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

for op ∈ (:push!, :pushposition!, :pushvelocity!)
  @eval function $(op)(plasma::Plasma, field::AbstractField, dt)
    for species ∈ plasma, particle ∈ species
      $(op)(particle, field, dt)
    end
    return plasma
  end
end

