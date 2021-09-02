using ElectrostaticPIC1D, JLD2, Random, Test; Random.seed!(0)

function go()
NG = 64 # number of grid points
NPPCPS = 16 # number of particles per cell per species
NS = 2 # number of species
L = 1.0 # length of domain
NP = NPPCPS * NG * NS # total number of particles
weight = 32π^2/3 * NPPCPS # /2 or *2? √2?
dde = 1
v0 = 1.0

run(`mkdir -p data`)

for (field, fieldtypename) ∈  ((FourierField(NG,L), "Fourier"),)
  run(`mkdir -p data/$fieldtypename`)
  for (particleshape, particletypename) ∈ ((DeltaFunctionShape(), "Delta"),
                                           (GaussianShape(L / NG), "Gaussian"),)
    nuclide = Nuclide(1.0, 1.0)
    
    leftpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    rightpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    
    left = Species([Particle(nuclide, particleshape; x=x, v=-v0, w=weight) for x in leftpositions])
    right = Species([Particle(nuclide, particleshape; x=x, v=v0, w=weight) for x in rightpositions])
    
    plasma = Plasma([left, right])
    @assert length(plasma) == NS
    @assert L == v0 == 1.0 # to make things simple for this particular test
    
    ti = LeapFrogTimeIntegrator(plasma, field)

    run(`mkdir -p data/$fieldtypename/$particletypename`)
    run(`rm -f data/$fieldtypename/$particletypename/"*".jld2`)
    stub = "data/$(fieldtypename)/$(particletypename)/"
    
    sim = Simulation(plasma, field, ti, diagnosticdumpevery=dde, endtime=1.0, filenamestub=stub)

    @test energy(sim.field) == 0.0 

    expectedenergy = weight * (mass(nuclide) * v0^2 / 2) * NP
    expectedmomentum = 0.0
    expectedchargedensity = weight * NP * charge(nuclide)

    run!(sim)

    for i in 1:dde:ElectrostaticPIC1D.diagnosticdumpcounter(sim)
      sim_i = ElectrostaticPIC1D.load(stub, i)
      particlecharge = charge(sim_i.plasma)
      fieldcharge = charge(sim_i.field)
      @test particlecharge ≈ expectedchargedensity
      @test fieldcharge ≈ expectedchargedensity
      particleenergy = energy(sim_i.plasma)
      fieldenergy = energy(sim_i.field)
      @test particleenergy + fieldenergy ≈ expectedenergy rtol=0.01
      particlemomentum = momentum(sim_i.plasma)
      @test particlemomentum ≈ expectedmomentum atol=sqrt(eps())
    end

  end
end

end

@testset "Simulations" begin
  go()
end
