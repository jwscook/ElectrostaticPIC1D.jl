
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

const INTEGRATOR_RTOL = 100eps()

@concrete struct SemiImplicitMidpointSingleLoopIntegrator <: AbstractTimeIntegrator
  dt::Float64
  atol::Float64
  rtol::Float64
  maxiters::Int64
  fieldcopy
end

function SemiImplicitMidpointSingleLoopIntegrator(p::Plasma, f::AbstractField;
    cflmultiplier=1, maxiters=16, atol=0.0, rtol=INTEGRATOR_RTOL)
  return SemiImplicitMidpointSingleLoopIntegrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f))
end

timestep(ti::SemiImplicitMidpointSingleLoopIntegrator) = ti.dt

function (ti::SemiImplicitMidpointSingleLoopIntegrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  copy!(ti.fieldcopy, field)
  firstiteration = true
  iters = 0
  while firstiteration || iters < ti.maxiters
    iters += 1

    zero!(field.chargedensity)

#    convergence = true

    for s ∈ plasma.species, particle ∈ s.particles
      if firstiteration
        empty!(particle.work)
        push!(particle.work, (position(particle), velocity(particle), 0.0)) # initial condition
        push!(particle.work, (position(particle), velocity(particle), 0.0)) # last iteration
      end
      velocity!(particle, (particle.work[1][2] + velocity(particle)) / 2) # set to midpoint velocity
      position!(particle, particle.work[1][1]) # set position to beginning of timestep
      pushposition!(particle, dt/2, bc) # push to midpoint with midpoint velocity
      deposit!(field, particle) # deposit charge for this iteration
      E = electricfield(field, particle) # read E field from last iteration
      pushposition!(particle, dt/2, bc) # second half push
      velocity!(particle, particle.work[1][2]) # reset velocity to start of iteration
      pushvelocity!(particle, E, dt) # accelerate with middle electricfield

#      convergence &= all(ispprox.(particle.work[1], (position(particle), velocity(particle)),
#                                  rtol=ti.rtol, atol=ti.atol))

      particle.work[2] = (position(particle), velocity(particle), E)
    end
    solve!(field) # solve for the mid point electric field

    if all(isapprox.(field.chargedensity, ti.fieldcopy.chargedensity, atol=ti.atol, rtol=ti.rtol))
#      @show iters, convergence, norm(field.chargedensity .- ti.fieldcopy.chargedensity)
      break
    end
    copy!(ti.fieldcopy.chargedensity, field.chargedensity)

    firstiteration = false
  end
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
    cflmultiplier=1, maxiters=16, atol=0.0, rtol=INTEGRATOR_RTOL)
  return SemiImplicit2ndOrderTimeIntegrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f))
end

timestep(ti::SemiImplicit2ndOrderTimeIntegrator) = ti.dt

function (ti::SemiImplicit2ndOrderTimeIntegrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  copy!(ti.fieldcopy, field)
  firstiteration = true
  iters = 0
  while firstiteration || iters < ti.maxiters
    iters += 1

    zero!(field.chargedensity)

    for species ∈ plasma, particle ∈ species
      if firstiteration
        empty!(particle.work)
        push!(particle.work, (position(particle), velocity(particle), 0.0)) # initial condition
      end
      velocity!(particle, (velocity(particle) + particle.work[1][2]) / 2) # set to half way velocity
      position!(particle, particle.work[1][1])
      pushposition!(particle, dt/2, bc)
      deposit!(field, particle)
    end
    solve!(field) # solve for the mid point electric field
    for species ∈ plasma, particle ∈ species
      E = electricfield(field, particle)
      pushposition!(particle, dt/2, bc) # second half push
      velocity!(particle, particle.work[1][2]) # set to starting velocity
      pushvelocity!(particle, E, dt) # accelerate with middle electricfield
    end

    if all(isapprox.(field.chargedensity, ti.fieldcopy.chargedensity, atol=ti.atol, rtol=ti.rtol))
      break
    end
    copy!(ti.fieldcopy.chargedensity, field.chargedensity)

    firstiteration = false
  end
  return nothing
end


@concrete struct SemiImplicitYoshida4Integrator <: AbstractTimeIntegrator
  dt::Float64
  atol::Float64
  rtol::Float64
  maxiters::Int64
  fieldcopy1
  fieldcopy2
  fieldcopy3
end

function SemiImplicitYoshida4Integrator(p::Plasma, f::AbstractField;
    cflmultiplier=1, maxiters=16, atol=0.0, rtol=INTEGRATOR_RTOL)
  return SemiImplicitYoshida4Integrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f), deepcopy(f), deepcopy(f))
end

timestep(ti::SemiImplicitYoshida4Integrator) = ti.dt

function (ti::SemiImplicitYoshida4Integrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  copy!(ti.fieldcopy1, field)
  copy!(ti.fieldcopy2, field)
  copy!(ti.fieldcopy3, field)
  firstiteration = true
  iters = 0
  d1 = d3 = w1 = 1 / (2 - cbrt(2))
  d2 = w0 = -cbrt(2) * w1
  c1 = c4 = w1 / 2
  c2 = c3 = (w0 + w1) / 2
  while firstiteration || iters < ti.maxiters
    iters += 1

    zero!(field.chargedensity)
    zero!(ti.fieldcopy1.chargedensity)
    zero!(ti.fieldcopy2.chargedensity)

    for species ∈ plasma, particle in species
      if firstiteration
        empty!(particle.work)
        push!(particle.work, (position(particle), velocity(particle), 0.0)) # initial condition
      end

      p = deepcopy(particle)
      position!(p, particle.work[1][1])
      velocity!(p, particle.work[1][2])

      pushposition!(p, c1 * dt, bc)
      deposit!(ti.fieldcopy1, p)
      pushvelocity!(p, ti.fieldcopy1, d1 * dt)

      pushposition!(p, c2 * dt, bc)
      deposit!(ti.fieldcopy2, p)
      pushvelocity!(p, ti.fieldcopy2, d2 * dt)

      pushposition!(p, c3 * dt, bc)
      deposit!(field, p)
      pushvelocity!(p, field, d3 * dt)

      position!(particle, position(p))
      velocity!(particle, velocity(p))
      pushposition!(particle, c4 * dt, bc)
    end
    solve!(field) # solve for the mid point electric field
    solve!(ti.fieldcopy1) # solve for the mid point electric field
    solve!(ti.fieldcopy2) # solve for the mid point electric field

    if all(isapprox.(field.chargedensity, ti.fieldcopy3.chargedensity, atol=ti.atol, rtol=ti.rtol))
      break
    end
    copy!(ti.fieldcopy3.chargedensity, field.chargedensity)

    firstiteration = false
  end
  return nothing
end

