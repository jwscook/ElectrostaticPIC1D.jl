
mutable struct SimulationParameters
  endtime::Float64
  filenamestub::String
  time::Float64
  iteration::Int
end

SimulationParameters(endtime, filenamestub) = SimulationParameters(endtime, filenamestub, 0.0, 0)

@concrete struct Simulation{F<:AbstractField}
  plasma::Plasma
  field::F
  timeintegrator
  diagnosticdumpevery::Int64
  endtime::Float64
  filenamestub::String
  time::Ref{Float64}
  iteration::Ref{Int}
end

function Simulation(plasma::Plasma, field::AbstractField, timeintegrator::AbstractTimeIntegrator;
  diagnosticdumpevery=1, endtime=10.0, filenamestub=nothing)
  isnothing(filenamestub) && (filenamestub = "$(typeof(eltype(plasma[1])))-$(typeof(field))-$(timeintegrator)-")
  return Simulation(plasma, field, timeintegrator, diagnosticdumpevery, endtime, filenamestub, Ref(0.0), Ref(0))
end
filenamestub(sim::Simulation) = sim.filenamestub
filename(sim::Simulation) = sim.filenamestub * "$(iteration(sim))" * ".jld2"
timestep(sim::Simulation) = timestep(sim.timeintegrator)
iteration(sim::Simulation) = sim.iteration[]
time(sim::Simulation) = sim.time[]
iterate!(sim::Simulation) = (sim.iteration[] += 1)
save(sim::Simulation) = @save "$(filename(sim))" sim

(sim::Simulation)() = run!(sim)

function run!(sim::Simulation)
  while time(sim) <= sim.endtime
    runtimestep!(sim)
  end
end

function runtimestep!(sim::Simulation)
  if mod(iteration(sim), sim.diagnosticdumpevery) == 0
    save(sim)
  end
  integrate!(sim.plasma, sim.field, sim.timeintegrator)
  @show sim.iteration[] += 1
  sim.time[] += timestep(sim.timeintegrator)
end

