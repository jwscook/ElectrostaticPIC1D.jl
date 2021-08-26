using ElectrostaticPIC1D, Random, Test; Random.seed!(0)

NG, NPPCPS, L = 64, 16, 1.0
field = FourierField(NG, L)

particleshape = DeltaFunctionShape()

nuclide = Nuclide(1.0, 1.0)
leftpositions = rand(NPPCPS) * L
rightpositions = rand(NPPCPS) * L

w = 32π^2/3 * NPPCPS # /2 or *2? √2?

left = Species([Particle(nuclide, particleshape; position=x, velocity=-1.0, weight=w) for x in leftpositions])
right = Species([Particle(nuclide, particleshape; position=x, velocity=1.0, weight=w) for x in rightpositions])

plasma = Plasma([left, right])

ti = LeapFrogTimeIntegrator(plasma, field)

sim = Simulation(plasma, field, ti, diagnosticdumpevery=1, endtime=10.0, filenamestub="data/")

run!(sim)



