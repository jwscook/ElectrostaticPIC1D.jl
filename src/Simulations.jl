
mutable struct SimulationParameters
  iteration::Int
  time::Float64
  endtime::Float64
  filenamestub::String
end

@concrete struct Simulation{F<:AbstractField}
  plasma::Plasma
  field::F
  timeintegrator
  diagnotictimestepinterval::Int64
  params::SimulationParameters
end
filenamestub(sim::Simulation) = sim.params.filenamestub
filename(sim::Simulation) = sim.params.filenamestub * "_" * "$(iteration(sim))" * ".jld2"
timestep(sim::Simulation) = timestep(sim.timeintegrator)
iteration(sim::Simulation) = sim.params.iteration
time(sim::Simulation) = sim.params.time
iterate!(sim::Simulation) = (sim.params.iteration += 1)
save(sim::Simulation) = @save "$(filename(sim))" sim

function integrate!(sim::Simulation)
  if mod(iteration(sim), sim.diagnotictimestepinterval) == 0
    save(sim)
  end
  integrate!(sim.plasma, sim.field, sim.timeintegrator)
  sim.params.iteration += 1
  sim.params.time += timestep(sim.timeintegrator)
end
function (sim::Simulation)()
  while time(sim) < sim.endtime
    integrate!(sim)
  end
  save(sim)
end

