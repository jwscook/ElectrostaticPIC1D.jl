
abstract type AbstractTimeIntegrator end

function integrate!(plasma, field::AbstractField, integrator::AbstractTimeIntegrator, dt=nothing)
  return integrator(plasma, field, dt)
end

struct LeapFropTimeIntegrator <: AbstractTimeIntegrator
  dt::Float64
end

function LeapFrogTimeIntegrator(p::Plasma, f::AbstractField)
  return LeapFropTimeIntegrator(cellsize(f) / maxspeed(p))
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
  plasmacopy
end

function SemiImplicit2ndOrderTimeIntegrator(p::Plasma, f::AbstractField; maxiters=10, atol=0, rtol=sqrt(eps()))
  return SemiImplicit2ndOrderTimeIntegrator(cellsize(f) / maxspeed(p), atol, rtol, maxiters, deepcopy(p), deepcopy(f))
end

timestep(ti::SemiImplicit2ndOrderTimeIntegrator) = ti.dt


function (ti::SemiImplicit2ndOrderTimeIntegrator)(plasma, field, dt=nothing)
  isnothing(dt) && (dt = timestep(ti))
  bc = BC(0.0, domainsize(field))
  zero!(field)
  fieldcopy = deepcopy(field) # TODO stop this from allocating, or figure how to not require a copy
  plasmacopy = deepcopy(plasma) # TODO stop this from allocating, or figure how to not require a copy
  firstiteration = true
  zero!(field) # field is to be known at the half timestep
  while firstiteration || !isapprox(field, fieldcopy, atol=ti.atol, rtol=ti.rtol) || iters < ti.maxiters
    firstiteration = false
    for (i,s) ∈ enumerate(plasma.species)
      for (j, particle) ∈ enumerate(s.particles)
        p₀ = plasmacopy[i][j]
        copy!(particle, p₀)
        pushposition!(particle, dt/2, bc) # push half with starting velocity
        pushvelocity!(particle, field, dt) # accelerate with middle velocity
        pushposition!(particle, dt/2, bc) # now push other half timestep with end timestep velocity
        deposit!(field, (p₀ + particle) / 2)
      end
    end
    fieldcopy = deepcopy(field) # copy across before solve
    solve!(field) # solve for the mid point electric field
  end
  return nothing
end

