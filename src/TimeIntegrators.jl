
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

function SemiImplicit2ndOrderTimeIntegrator(p::Plasma, f::AbstractField; maxiters=10, atol=0, rtol=sqrt(eps()))
  return SemiImplicit2ndOrderTimeIntegrator(cellsize(f) / maxspeed(p), atol, rtol, maxiters, deepcopy(p))
end

timestep(ti::SemiImplicit2ndOrderTimeIntegrator) = ti.dt


function (ti::SemiImplicit2ndOrderTimeIntegrator)(plasma, field, dt=nothing)
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(0.0, domainsize(field))
  copy!(ti.fieldcopy, field)
  firstiteration = true
  while firstiteration || iters < ti.maxiters
    firstiteration = false
    zero!(field.chargedensity)
    for (i,s) ∈ enumerate(plasma.species)
      for (j, particle) ∈ enumerate(s.particles)
        p₀ = deepcopy(particle)
        copy!(particle, p₀)
        pushposition!(particle, dt/2, bc) # push half with starting velocity
        pushvelocity!(particle, field, dt) # accelerate with middle velocity
        pushposition!(particle, dt/2, bc) # now push other half timestep with end timestep velocity
        deposit!(field, (p₀ + particle) / 2)
      end
    end
    solve!(field) # solve for the mid point electric field
    if isapprox(field, fieldcopy, atol=ti.atol, rtol=ti.rtol) 
      pushposition!(plasma, dt / 2, bc)
      pushvelocity!(plasma, field, dt)
      pushposition!(plasma, dt / 2, bc)
      break
    end
    copy!(ti.fieldcopy, field)
  end
  return nothing
end

