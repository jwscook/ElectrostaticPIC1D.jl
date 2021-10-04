
abstract type AbstractTimeIntegrator end

function integrate!(plasma, field::AbstractField, integrator::AbstractTimeIntegrator, dt=nothing)
  return integrator(plasma, field, dt)
end

struct LeapFropTimeIntegrator <: AbstractTimeIntegrator
  dt::Float64
end

function LeapFrogTimeIntegrator(p::Plasma, f::AbstractField; cflmultiplier=1)
  return LeapFropTimeIntegrator(cellsize(f) / maxspeed(p) * cflmultiplier)
end

timestep(ti::LeapFropTimeIntegrator) = ti.dt

function (ti::LeapFropTimeIntegrator)(plasma, field::AbstractField{BC}, dt=nothing) where {BC}
  isnothing(dt) && (dt = timestep(ti))

  bc = BC(0.0, domainsize(field))

  pushposition!(plasma, dt / 2, bc)
  zero!(field)
  deposit!(field, plasma)
  solve!(field)
  pushvelocity!(plasma, field, dt)
  pushposition!(plasma, dt / 2, bc)
  return nothing
end


@concrete struct SemiImplicit2ndOrderTimeIntegrator <: AbstractTimeIntegrator
  dt::Float64
  atol::Float64
  rtol::Float64
  maxiters::Int64
  fieldcopy
end

function SemiImplicit2ndOrderTimeIntegrator(p::Plasma, f::AbstractField;
    cflmultiplier=1, maxiters=10, atol=0.0, rtol=sqrt(eps()))
  return SemiImplicit2ndOrderTimeIntegrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f))
end

timestep(ti::SemiImplicit2ndOrderTimeIntegrator) = ti.dt

function (ti::SemiImplicit2ndOrderTimeIntegrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  copy!(ti.fieldcopy, field)
  plasma0 = deepcopy(plasma) # TODO - figure out how to not use a copy
  firstiteration = true
  iters = 0
  while firstiteration || iters < ti.maxiters
    iters += 1
    firstiteration = false

    zero!(field.chargedensity)

    for (s0,s) ∈ zip(plasma0.species, plasma.species)
      for (p0, particle) ∈ zip(s0.particles, s.particles)
        v0 = particle.velocity
        particle.velocity = (particle.velocity + p0.velocity) / 2 # set to half way velocity
        particle.basis.centre = p0.basis.centre
        pushposition!(particle, dt/2, bc)
        deposit!(field, particle)
      end
    end
    solve!(field) # solve for the mid point electric field
    for (s0,s) ∈ zip(plasma0.species, plasma.species)
      for (p0, particle) ∈ zip(s0.particles, s.particles)
        E = electricfield(field, particle)
        pushposition!(particle, dt/2, bc) # second half push
        particle.velocity = p0.velocity
        pushvelocity!(particle, E, dt) # accelerate with middle electricfield
      end
    end

    if isapprox(field.chargedensity, ti.fieldcopy.chargedensity, atol=ti.atol, rtol=ti.rtol)
      break
    end
    copy!(ti.fieldcopy.chargedensity, field.chargedensity)
  end
  return nothing
end


