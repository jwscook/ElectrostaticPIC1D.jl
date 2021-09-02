using ElectrostaticPIC1D, JLD2, Random, Test; Random.seed!(0)

function go()
NG = 64
NPPCPS = 16
L = 1.0
weight = 32π^2/3 * NPPCPS # /2 or *2? √2?
dde = 10

run(`mkdir -p data`)

for (field, fieldtypename) ∈  ((FourierField(NG,L), "Fourier"),)
  run(`mkdir -p data/$fieldtypename`)
  for (particleshape, particletypename) ∈ ((DeltaFunctionShape(), "Delta"),
                                           (GaussianShape(L / NG), "Gaussian"),)
    nuclide = Nuclide(1.0, 1.0)
    
    leftpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    rightpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    
    left = Species([Particle(nuclide, particleshape; x=x, v=-1.0, w=weight) for x in leftpositions])
    right = Species([Particle(nuclide, particleshape; x=x, v=1.0, w=weight) for x in rightpositions])
    
    plasma = Plasma([left, right])
    
    ti = LeapFrogTimeIntegrator(plasma, field)

    run(`mkdir -p data/$fieldtypename/$particletypename`)
    run(`rm -f data/$fieldtypename/$particletypename/"*".jld2`)
    stub = "data/$(fieldtypename)/$(particletypename)/"
    
    sim = Simulation(plasma, field, ti, diagnosticdumpevery=dde, endtime=10.0, filenamestub=stub)

    run!(sim)

    for i in 1:dde:ElectrostaticPIC1D.diagnosticdumpcounter(sim)
      sim_i = ElectrostaticPIC1D.load(stub, i)
      expectedchargedensity = weight * NPPCPS * NG * 2
      particlecharge = charge(sim_i.plasma)
      fieldcharge = charge(sim_i.field)
      @test particlecharge ≈ expectedchargedensity
      @test fieldcharge ≈ expectedchargedensity
    end

  end
end

end

@testset "Simulations" begin
  go()
end
