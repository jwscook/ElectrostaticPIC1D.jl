using ElectrostaticPIC1D, JLD2, Plots, Random, Test; Random.seed!(0)

function go()
NG = 64 # number of grid points
NPPCPS = 16 # number of particles per cell per species
NS = 2 # number of species
L = 1.0 # length of domain
NP = NPPCPS * NG * NS # total number of particles
q = m = 1.0
v0 = 1.0
nvortexpergrid = 1
plasmafrequency = 2/√3 * v0 * nvortexpergrid * 2π/L
# division by √2 because this is plasma frequency for a single species
density = plasmafrequency^2 / q^2 * m
weight = density * L / (NPPCPS * NG) * 1e2
normalisedgrowthrate(x) = imag(sqrt(Complex(x^2+1-sqrt(4*x^2+1))))
wavenumbers = 2*pi/L * (1:NG/2)
normalisedgamma, fastestgrowingmode = findmax(normalisedgrowthrate.(v0 .* wavenumbers / plasmafrequency))
growthrate = normalisedgamma * plasmafrequency
@show fastestgrowingmode, growthrate

nuclide = Nuclide(q, m)
dde = 20

run(`rm -rf data`)
run(`mkdir -p data`)

Δ = L / NG

particleshapes = ((DeltaFunctionShape(), "Delta"),
                  (GaussianShape(Δ), "Gaussian"),
                  (BSpline{0}(Δ), "BSpline0"),
                  (BSpline{1}(Δ), "BSpline1"),
                  (BSpline{2}(Δ), "BSpline2"),)
                  
fieldsolvers = ((FourierField(NG,L), "Fourier"),
                (LSFEMField(NG,L,BSpline{1}(Δ)), "LSFEM_BSpline1"),
                (LSFEMField(NG,L,BSpline{2}(Δ)), "LSFEM_BSpline2"),
                (LSFEMField(NG,L,GaussianShape(Δ * √2)), "LSFEM_Gaussian"),
                (GalerkinFEMField(NG,L,BSpline{0}(Δ), BSpline{1}(Δ)), "Galerkin_BSpline0_BSpline1"),
                (GalerkinFEMField(NG,L,BSpline{1}(Δ), BSpline{2}(Δ)), "Galerkin_BSpline1_BSpline2"),
                (FiniteDifferenceField(NG,L,order=1), "FiniteDifference1"),
                (FiniteDifferenceField(NG,L,order=2), "FiniteDifference2"),
                (FiniteDifferenceField(NG,L,order=4), "FiniteDifference4"),)

for (particleshape, particletypename) ∈ particleshapes
  run(`mkdir -p data/$particleshape`)
  for (field, fieldtypename) ∈  fieldsolvers
    leftpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    rightpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
    budge(x) = rand(Bool) ? prevfloat(x) : nextfloat(x)
    leftpositions .= budge.(leftpositions)
    rightpositions .= budge.(rightpositions)
    
    left = Species([Particle(nuclide, particleshape; x=x, v=-v0, w=weight) for x in leftpositions])
    right = Species([Particle(nuclide, particleshape; x=x, v=v0, w=weight) for x in rightpositions])
    
    plasma = Plasma([left, right])
    @assert length(plasma) == NS
    @assert L == v0 == 1.0 # to make things simple for this particular test
    
    ti = LeapFrogTimeIntegrator(plasma, field; cflmultiplier=1/pi^2)

    run(`mkdir -p data/$particletypename/$fieldtypename`)
    stub = "data/$(particletypename)/$(fieldtypename)/"
    
    sim = Simulation(plasma, field, ti, diagnosticdumpevery=dde, endtime=10.0, filenamestub=stub)

    #@test energydensity(sim.field) == 0.0 

    expectedenergy = weight * (mass(nuclide) * v0^2 / 2) * NP
    expectedmomentum = 0.0
    expectedchargedensity = weight * NP * charge(nuclide)
    singlepseudoparticlemomentum = v0 * mass(nuclide) * weight
    electricfieldnormalisation =  expectedchargedensity * L

    run!(sim)

    x = cellcentres(field)
    fileindices = 0:ElectrostaticPIC1D.diagnosticdumpcounter(sim)
    fieldenergies = []
    particleenergies = []
    particlemomenta = []
    times = []
    #try
      anim = @animate for i in fileindices
        sim_i = ElectrostaticPIC1D.load(stub, i)
        particlecharge = chargedensity(sim_i.plasma)
        fieldcharge = chargedensity(sim_i.field)
        particleenergy = energydensity(sim_i.plasma)
        fieldenergy = energydensity(sim_i.field)
        particlemomentum = momentumdensity(sim_i.plasma)
        push!(times, sim_i.time[])
        #@test particlecharge ≈ expectedchargedensity
        #@test fieldcharge ≈ expectedchargedensity
        #@test particleenergy + fieldenergy ≈ expectedenergy rtol=0.01
        #@test particlemomentum ≈ expectedmomentum atol=sqrt(eps()) rtol=1/NP

        scatter(positions(sim_i.plasma), velocities(sim_i.plasma), label=nothing)
        #plot!(x, sim_i.field.electricfield.(x) ./ electricfieldnormalisation, label="E")
        #plot!(x, sim_i.field.charge.(x) ./ expectedchargedensity, label="ρ")
        push!(fieldenergies, fieldenergy)
        push!(particleenergies, particleenergy)
        push!(particlemomenta, particlemomentum)
        #plot!((1:i) / maximum(fileindices), fieldenergies / expectedenergy, label="field energy")
        #plot!((1:i) / maximum(fileindices), particleenergies / expectedenergy, label="particle energy")
        #plot!((1:i) / maximum(fileindices), particlemomenta / singlepseudoparticlemomentum, label="momentum")
        #xlabel!("Position")
        #ylabel!("Velocity")
        xlims!(0, L)
        ylims!(-3, 3)
      end
      gif(anim, stub * "animation.gif")
      @save "plotdata.jld2" fieldenergies expectedenergy times growthrate

      y = fieldenergies / expectedenergy
      plot(times, log10.(y), label="field energy")
      index = findfirst(y .> 1e-2)
      index = isnothing(index) ? 1 : index
      plot!(times, 2*growthrate / log(10) .* (times .- times[index]) .+ log10.(abs.(y[index])),label="predicted")

      y = particlemomenta ./ singlepseudoparticlemomentum
      plot!(times, log10.(abs.(y .- y[1])), label="momentum")
      y = (fieldenergies .+ particleenergies) / expectedenergy
      plot!(times, log10.(abs.(y .- y[1])), label="energy error")

      xlabel!("Time", legend=:bottomright)
      xlims!(0, maximum(times))
      ylims!(-35, 1.0)
      savefig(stub * "plot.pdf")
    #catch
    #end
  end
end

end

@testset "Simulations" begin
  go()
end
