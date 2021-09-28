using Distributed

addprocs(Sys.CPU_THREADS ÷ 2)

@everywhere using ElectrostaticPIC1D, FFTW, JLD2, Plots, Random, Test, ThreadsX
@everywhere Random.seed!(0)

@everywhere function go(;NG=64, NPPCPS=16, L=1.0, nvortexpergrid=1,
    cleandatadir=true, dryrunabort=false, datadir="data")
#  NG = 64 # number of grid points
#  NPPCPS = 16 # number of particles per cell per species
#  nvortexpergrid = 1 # number of vortices per grid, should really be an Int
#  L = 1.0 # length of domain

  NS = 2 # number of species
  NP = NPPCPS * NG * NS # total number of particles
  q = m = 1.0 # charge, mass
  v0 = 1.0 # particle speed
  plasmafrequency = 2/√3 * v0 * nvortexpergrid * 2π/L
  density = plasmafrequency^2 / q^2 * m # plasma number density per species
  weight = density * L / (NPPCPS * NG) # weight of each particle
  analyticresult(x; op, sign) = op(sqrt(Complex(x^2+1 + sign * sqrt(4*x^2+1))))
  normalisedgrowthrate(x) = analyticresult(x; op=imag, sign=-1)
  wavenumbers = 2*pi/L * (1:NG/2)
  normalisedgamma, fastestgrowingmode = findmax(normalisedgrowthrate.(v0 .* wavenumbers / plasmafrequency))
  growthrate = normalisedgamma * plasmafrequency

  nuclide = Nuclide(q, m) # the nuclide for the test

  cleandatadir && run(`rm -rf $datadir`)

  Δ = L / NG

  particleshapes = ((DeltaFunctionShape(), "Delta"),
                    (GaussianShape(Δ), "Gaussian"),
                    #(BSpline{0}(Δ), "BSpline0"),
                    #(BSpline{1}(Δ), "BSpline1"),
                    #(BSpline{2}(Δ), "BSpline2"),
                    )

  fieldsolvers = ((FourierField(NG,L), "Fourier"),
                  (LSFEMField(NG,L,BSpline{1}(Δ)), "LSFEM_BSpline1"),
                  (LSFEMField(NG,L,BSpline{2}(Δ)), "LSFEM_BSpline2"),
                  (LSFEMField(NG,L,GaussianShape(Δ * √2)), "LSFEM_Gaussian"),
                  (GalerkinFEMField(NG,L,BSpline{1}(Δ), BSpline{2}(Δ)), "Galerkin_BSpline1_BSpline2"),
                  #(FiniteDifferenceField(NG,L,order=1), "FiniteDifference1"),
                  #(FiniteDifferenceField(NG,L,order=2), "FiniteDifference2"),
                  #(FiniteDifferenceField(NG,L,order=4), "FiniteDifference4"),
                  )

  particleics = Dict()

  leftpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L
  rightpositions = mod.((bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64 .+ rand(), 1) * L

  #particleics[:Quiet] = deepcopy((leftpositions, rightpositions))

  budge(x) = x + (rand() - 0.5) * Δ / NPPCPS / 10;

  particleics[:Muffled] = deepcopy((budge.(leftpositions), budge.(rightpositions)))

  #particleics[:Noisey] = (rand(NPPCPS * NG) .* L, rand(NPPCPS * NG) .* L)

  #particleics[:Seeded] = (leftpositions*(1-p) + p*(asin.(2leftpositions-1)/π+0.5),
  #                        rightpositions*(1-p) - p*(asin.(2rightpositions-1)/π+0.5))

  function innerloop(inputs)

    particleshape, particletypename = inputs[1]

    field, fieldtypename = inputs[2]

    icname, particlepositions = inputs[3]
    xl, xr = particlepositions # x-left, x-right

    run(`mkdir -p $datadir/$fieldtypename/$particletypename/$icname`)
    stub = "$(datadir)/$(fieldtypename)/$(particletypename)/$(icname)/"

    left = Species([Particle(nuclide, particleshape; x=x, v=-v0, w=weight) for x in mod.(xl, L)])
    right = Species([Particle(nuclide, particleshape; x=x, v=v0, w=weight) for x in mod.(xr, L)])

    plasma = Plasma([left, right])

    @assert length(plasma) == NS
    @assert L == v0 == 1.0 # to make things simple for this particular test

    ti = LeapFrogTimeIntegrator(plasma, field; cflmultiplier=1/7)

    expectedenergy = weight * (mass(nuclide) * v0^2 / 2) * NP
    expectedmomentum = 0.0
    expectedchargedensity = weight * NP * charge(nuclide)
    singlepseudoparticlemomentum = v0 * mass(nuclide) * weight
    electricfieldnormalisation = expectedchargedensity * L / NG

    x = cellcentres(field)
    dryrunabort && return nothing


    try
      sim = Simulation(plasma, field, ti, diagnosticdumpevery=10,
        endtime=15.0, filenamestub=stub)

      init!(sim) # set up to start
      @time run!(sim)

      fileindices = 0:ElectrostaticPIC1D.diagnosticdumpcounter(sim)
      Ex = zeros(length(x), length(fileindices))
      Rho = zeros(length(x), length(fileindices))
      fieldenergies = []
      particleenergies = []
      particlemomenta = []
      times = []

      anim = @animate for (i, fileindex) in enumerate(fileindices)
        sim_i = ElectrostaticPIC1D.load(stub, fileindex)

        push!(times, sim_i.time[])
        scatter(positions(sim_i.plasma), velocities(sim_i.plasma), label=nothing)

        #particlecharge = charge(sim_i.plasma)
        #fieldcharge = chargedensity(sim_i.field) * L
        particleenergy = energy(sim_i.plasma)
        fieldenergy = energydensity(sim_i.field) * L
        particlemomentum = momentum(sim_i.plasma)
        #@show minimum(sim_i.field.charge)
        #@show maximum(sim_i.field.charge)
        #@show minimum(sim_i.field.electricfield)
        #@show maximum(sim_i.field.electricfield)
        #@test particlecharge ≈ expectedchargedensity
        #@test fieldcharge ≈ expectedchargedensity
        #@test particleenergy + fieldenergy ≈ expectedenergy rtol=0.01
        #@test particlemomentum ≈ expectedmomentum atol=sqrt(eps()) rtol=1/NP

        Ex[:, i] .= sim_i.field.electricfield.(x) ./ electricfieldnormalisation
        Rho[:, i] .= sim_i.field.chargedensity.(x) ./ expectedchargedensity
        plot!(x, sim_i.field.electricfield.(x) ./ electricfieldnormalisation, label="E")
        plot!(x, sim_i.field.chargedensity.(x) ./ expectedchargedensity, label="ρ")
        push!(fieldenergies, fieldenergy)
        push!(particleenergies, particleenergy)
        push!(particlemomenta, particlemomentum)
        #plot!((1:i) / maximum(fileindices), fieldenergies / expectedenergy, label="field energy")
        #plot!((1:i) / maximum(fileindices), particleenergies / expectedenergy, label="particle energy")
        #plot!((1:i) / maximum(fileindices), particlemomenta / singlepseudoparticlemomentum, label="momentum")
        xlabel!("Position")
        ylabel!("Velocity")
        annotate!(0.05, -2.95, text("$(trunc(sim_i.time[], digits=3))", :left))
        xlims!(0, L)
        ylims!(-3, 3)
        annotate!(0.02, 2.9, text("p=$particletypename", :left))
        annotate!(0.02, 2.6, text("f=$fieldtypename", :left))
      end
      gif(anim, stub * "animation.gif")
      @save stub * "energydata.jld2" fieldenergies particleenergies expectedenergy times growthrate
      @save stub * "momentumdata.jld2" particlemomenta singlepseudoparticlemomentum times

      y = fieldenergies / expectedenergy
      plot(times, log10.(y), label="field energy")
      index = findfirst(y .> 1e-2)
      index = isnothing(index) ? 10 : index
      plot!(times, 2*growthrate / log(10) .* (times .- times[index]) .+ log10.(abs.(y[index])),label="predicted")

      y = particlemomenta ./ singlepseudoparticlemomentum
      plot!(times, log10.(abs.(y .- y[1])), label="momentum")
      y = (fieldenergies .+ particleenergies) / expectedenergy
      plot!(times, log10.(abs.(y .- y[1])), label="energy error")

      xlabel!("Time", legend=:bottomright)
      xlims!(0, maximum(times))
      ylims!(-17, 1.0)
      savefig(stub * "plot.pdf")

      for (Z, fieldstring) ∈ ((Ex, "Ex"), (Rho, "Rho"))
        LogAbsFFTEx = log10.(abs.(fft(Z')));
        LogAbsFFTEx[:, 1] *= NaN; # zeros on the log10 colorbar destroy the contrast
        LogAbsFFTEx[:, (end÷2) + 1 ] *= NaN;
        ks = (-NG÷2 : NG÷2 - 1)
        ws = (0:length(times)÷2) ./ times[end] / (plasmafrequency / 2π)
        heatmap(ks, ws, fftshift(LogAbsFFTEx)[(end÷2)+1:end, :])
        xlabel!("Wavenumber [Waves per box]")
        ylabel!("Frequency [Π]")
        plot!(ks, analyticresult.(v0 .* ks * (2π/L) / plasmafrequency; op=imag, sign=-1), label="")
        plot!(ks, analyticresult.(v0 .* ks * (2π/L) / plasmafrequency; op=real, sign=-1), label="")
        plot!(ks, analyticresult.(v0 .* ks * (2π/L) / plasmafrequency; op=real, sign=+1), label="")
        ylims!(0, maximum(ws))
        xlims!(ks[1], ks[end])
        savefig(stub * "$(fieldstring)_dispersionrelation.pdf")
      end
      @save stub * "dispersionrelation.jld2" Ex Rho times x L plasmafrequency

    catch e
      @show fieldtypename, particletypename
      @show e
      rethrow(e)
    end

    return nothing
  end

  inputs = Iterators.product(particleshapes, fieldsolvers, particleics)

  pmap(innerloop, inputs)

end

@testset "Simulations" begin
  go(; datadir="data_twostream")
end
