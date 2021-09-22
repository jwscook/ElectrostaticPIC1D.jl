
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
  diagnosticdumpcounter::Ref{Int}
end

function Simulation(plasma::Plasma, field::AbstractField, timeintegrator::AbstractTimeIntegrator;
  diagnosticdumpevery=1, endtime=10.0, filenamestub=nothing)
  isnothing(filenamestub) && (filenamestub = "$(typeof(eltype(plasma[1])))-$(typeof(field))-$(timeintegrator)-")
  return Simulation(plasma, field, timeintegrator, diagnosticdumpevery, endtime, filenamestub, Ref(0.0), Ref(0), Ref(-1))
end
filenamestub(sim::Simulation) = sim.filenamestub
filename(filepathstub::String, i::Int) = filepathstub * lpad("$i", 6, "0") * ".jld2"
filename(sim::Simulation) = filename(sim.filenamestub, diagnosticdumpcounter(sim))
timestep(sim::Simulation) = timestep(sim.timeintegrator)
iteration(sim::Simulation) = sim.iteration[]
diagnosticdumpcounter(sim::Simulation) = sim.diagnosticdumpcounter[]
time(sim::Simulation) = sim.time[]
iterate!(sim::Simulation) = (sim.iteration[] += 1)
function save(sim::Simulation)
  sim.diagnosticdumpcounter[] += 1
  @save "$(filename(sim))" sim
end
function load(filepathstub::String, diagnosticdumpcount::Int)
  @load "$(filename(filepathstub, diagnosticdumpcount))" sim
  return sim
end

function init!(sim::Simulation)
  sim.diagnosticdumpcounter[] = -1
  zero!(sim.field)
  deposit!(sim.field, sim.plasma)
  solve!(sim.field)
  return sim
end
function run!(sim::Simulation)
  save(sim) # save initial conditions as file 0
  while time(sim) <= sim.endtime
    runtimestep!(sim)
  end
end

function runtimestep!(sim::Simulation)
  integrate!(sim.plasma, sim.field, sim.timeintegrator)
  sim.iteration[] += 1
  sim.time[] += timestep(sim.timeintegrator)
  if mod(iteration(sim), sim.diagnosticdumpevery) == 0
    save(sim)
  end
  return sim
end

