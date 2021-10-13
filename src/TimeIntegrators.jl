
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

const INTEGRATOR_RTOL = 2eps()

@concrete struct SemiImplicitMidpointSingleLoopIntegrator <: AbstractTimeIntegrator
  dt::Float64
  atol::Float64
  rtol::Float64
  maxiters::Int64
  initfield
  fieldcopy
end

function SemiImplicitMidpointSingleLoopIntegrator(p::Plasma, f::AbstractField;
    cflmultiplier=1, maxiters=64, atol=0.0, rtol=INTEGRATOR_RTOL)
  return SemiImplicitMidpointSingleLoopIntegrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f), deepcopy(f))
end

timestep(ti::SemiImplicitMidpointSingleLoopIntegrator) = ti.dt

function (ti::SemiImplicitMidpointSingleLoopIntegrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  copy!(ti.initfield, field)
  copy!(ti.fieldcopy, field)

  solve!(ti.initfield)
  firstiteration = true
  iters = 0
#  maxsubiter = 0
  while firstiteration || iters < ti.maxiters
    iters += 1

    zero!(field.chargedensity)

    for s ∈ plasma.species, particle ∈ s.particles
      if firstiteration
        empty!(particle.work)
        E⁰ = electricfield(ti.initfield, particle)
        push!(particle.work, (position(particle), velocity(particle), E⁰)) # initial condition
      end

      x⁰, v⁰, E⁰ = particle.work[1]
      v¹ = velocity(particle)
      firstsubiteration = true
      subiters = 0
      E¹ = E⁰
      while (subiters += 1) < ti.maxiters
        Eᵇ = E¹
        E¹ = electricfield(field, particle)
        velocity!(particle, v⁰) # set to starting velocity
        pushvelocity!(particle, (E⁰ + E¹)/2, dt) # end time velocity
        v¹ = velocity(particle)
        position!(particle, x⁰)
        velocity!(particle, (v¹ + v⁰)/2) # set to mid point velocity
        pushposition!(particle, dt, bc) # end time velocity
        velocity!(particle, v¹) # set to end point velocity
        firstsubiteration = false
        isapprox(E¹, Eᵇ, atol=0.0, rtol=ti.rtol) && break
      end
#      maxsubiter = max(maxsubiter, subiters)
      deposit!(field, particle)
    end
    solve!(field) # solve for the end point electric field

    all(field.chargedensity .== ti.fieldcopy.chargedensity) && break
    if all(isapprox.(field.chargedensity, ti.fieldcopy.chargedensity, atol=ti.atol, rtol=ti.rtol))
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
    cflmultiplier=1, maxiters=64, atol=0.0, rtol=INTEGRATOR_RTOL)
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
    cflmultiplier=1, maxiters=64, atol=0.0, rtol=INTEGRATOR_RTOL)
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
    solve!(field)
    solve!(ti.fieldcopy1)
    solve!(ti.fieldcopy2)

    if all(isapprox.(field.chargedensity, ti.fieldcopy3.chargedensity, atol=ti.atol, rtol=ti.rtol))
      break
    end
    copy!(ti.fieldcopy3.chargedensity, field.chargedensity)

    firstiteration = false
  end
  return nothing
end

@concrete struct SemiImplicitSimpson13Integrator <: AbstractTimeIntegrator
  dt::Float64
  atol::Float64
  rtol::Float64
  maxiters::Int64
  fieldcopy1
  fieldcopy2
  fieldcopy3
end

function SemiImplicitSimpson13Integrator(p::Plasma, f::AbstractField;
    cflmultiplier=1, maxiters=64, atol=0.0, rtol=INTEGRATOR_RTOL)
  return SemiImplicitSimpson13Integrator(cellsize(f) / maxspeed(p) * cflmultiplier,
    atol, rtol, maxiters, deepcopy(f), deepcopy(f), deepcopy(f))
end

timestep(ti::SemiImplicitSimpson13Integrator) = ti.dt

function (ti::SemiImplicitSimpson13Integrator)(plasma, field::AbstractField{BC}, dt=nothing
    ) where {BC}
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(domainsize(field))
  solve!(field)
  copy!(ti.fieldcopy1, field)
  copy!(ti.fieldcopy2, field)
  firstiteration = true
  iters = 0
  while firstiteration || iters < ti.maxiters
    iters += 1

    zero!(field.chargedensity)
    zero!(ti.fieldcopy1.chargedensity)
    zero!(ti.fieldcopy2.chargedensity)


    for species ∈ plasma, particle in species
      if firstiteration
        empty!(particle.work)
        push!(particle.work, (position(particle), velocity(particle), 0.0)) # initial condition
        pushposition!(particle, dt, bc) # take a good guess at where it should end up
      end

      x = position(particle) # latest guess at end point position

      velocity!(particle, particle.work[1][2]) # set to starting velocity
      position!(particle, particle.work[1][1]) # set to starting position
      pushvelocity!(particle, ti.fieldcopy1, dt / 6) # 1/6 from E at start of timestep
      position!(particle, (x + particle.work[1][1]) / 2) # mid point position
      deposit!(ti.fieldcopy2, particle) # deposit half way X
      pushvelocity!(particle, ti.fieldcopy2, 4 * dt / 6) # 2/3 from E at mid point
      position!(particle, x) # set to end position
      deposit!(field, particle)
      pushvelocity!(particle, field, dt / 6) # 1/6 from E at end of timestep

      v = velocity(particle) # latest guess at end point velocity
      velocity!(particle, (velocity(particle) + particle.work[1][2]) / 2)
      position!(particle, particle.work[1][1])
      pushposition!(particle, dt, bc)
      velocity!(particle, v)
    end
    solve!(ti.fieldcopy2) # solve for the mid point electric field
    solve!(field) # solve for the end point

    if !firstiteration && all(isapprox.(field.chargedensity, ti.fieldcopy3.chargedensity, atol=ti.atol, rtol=ti.rtol))
      break
    end
    copy!(ti.fieldcopy3.chargedensity, field.chargedensity)

    firstiteration = false
  end
  return nothing
end

