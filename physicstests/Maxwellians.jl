using Distributed

addprocs(2)
#addprocs(Sys.CPU_THREADS ÷ 2)

@everywhere using ElectrostaticPIC1D, FFTW, JLD2, Plots, Random, SpecialFunctions, Test, ThreadsX
@everywhere Random.seed!(0)

@everywhere function go(;NG=64, NPPCPS=16, L=1.0, nvortexpergrid=1,
    cleandatadir=true, dryrunabort=false, datadir="data")
#  NG = 64 # number of grid points
#  NPPCPS = 16 # number of particles per cell per species
#  nvortexpergrid = 1 # number of vortices per grid, should really be an Int
#  L = 1.0 # length of domain

  Δ = L / NG
  NS = 2 # number of species
  NP = NPPCPS * NG * NS # total number of particles
  q0 = me = T = 1 # electron charge, mass and temperature are all unity
  # unit temperature, unit electron mass, so Debye length is 2π / Πₑ
  # Set Debye length = Δ, so Πₑ = 1 / Δ
  electronplasmafrequency = sqrt(T / me) / Δ
  density = electronplasmafrequency^2 / q0^2 * me # plasma number density for electrons
  weight = density * L / (NPPCPS * NG) # weight of each particle
  mi = 1836 * 2 * me

  electronnuclide = Nuclide(-q0, me) # electron
  ionnuclide = Nuclide(q0, mi) # ion

  cleandatadir && run(`rm -rf $datadir`)

  particleshapes = ((DeltaFunctionShape(), "Delta"),
                    (GaussianShape(Δ), "Gaussian"),
                    #(BSpline{0}(Δ), "BSpline0"),
                    #(BSpline{1}(Δ), "BSpline1"),
                    #(BSpline{2}(Δ), "BSpline2"),
                    )

  fieldsolvers = (#(FourierField(NG,L), "Fourier"),
                  (LSFEMField(NG,L,BSpline{1}(Δ)), "LSFEM_BSpline1"),
                  (LSFEMField(NG,L,BSpline{2}(Δ)), "LSFEM_BSpline2"),
                  (LSFEMField(NG,L,GaussianShape(Δ * √2)), "LSFEM_Gaussian"),
                  #(GalerkinFEMField(NG,L,BSpline{1}(Δ), BSpline{2}(Δ)), "Galerkin_BSpline1_BSpline2"),
                  #(FiniteDifferenceField(NG,L,order=1), "FiniteDifference1"),
                  #(FiniteDifferenceField(NG,L,order=2), "FiniteDifference2"),
                  #(FiniteDifferenceField(NG,L,order=4), "FiniteDifference4"),
                  )

  particleics = Dict()

  x1 = (bitreverse.(0:NPPCPS * NG-1) .+ 2.0^63) / 2.0^64
  xe = mod.(x1, L)
  xi = mod.(x1, L)
  ve = erfinv.(2 .* mod.(x1 .+ π, 1) .- 1) .* √(T / me)
  vi = erfinv.(2 .* mod.(x1 .+ π, 1) .- 1) .* √(T / mi)
  @assert all(isfinite, xe)
  @assert all(isfinite, xi)
  @assert all(isfinite, ve)
  @assert all(isfinite, vi)

  particleics[:Quiet] = deepcopy((xe, ve, xi, vi))

  function innerloop(inputs)

    particleshape, particletypename = inputs[1]

    field, fieldtypename = inputs[2]

    icname, particleics = inputs[3]
    xe, ve, xi, vi = particleics

    run(`mkdir -p $datadir/$fieldtypename/$particletypename/$icname`)
    stub = "$(datadir)/$(fieldtypename)/$(particletypename)/$(icname)/"

    electrons = Species([Particle(electronnuclide, particleshape; x=xe[i], v=ve[i], w=weight)
                    for i in eachindex(xe)])
    ions = Species([Particle(ionnuclide, particleshape; x=xi[i], v=vi[i], w=weight)
                    for i in eachindex(xi)])

    plasma = Plasma([electrons, ions])

    ti = LeapFrogTimeIntegrator(plasma, field; cflmultiplier=1/2)

    expectedenergy = weight * NP / 2
    expectedmomentum = 0.0
    @show chargedensitynormalisation = weight * NP * (abs(charge(electronnuclide)) + abs(charge(ionnuclide)))
    @show electricfieldnormalisation = chargedensitynormalisation * L / NG

    x = cellcentres(field)
    dryrunabort && return nothing


    try
      sim = Simulation(plasma, field, ti, endtime=10.0, filenamestub=stub)

      init!(sim) # set up to start
      @time run!(sim)

      fileindices = 0:ElectrostaticPIC1D.diagnosticdumpcounter(sim)
      Ex = zeros(length(x), length(fileindices))
      Rho = zeros(length(x), length(fileindices))
      fieldenergies = []
      particleenergies = []
      particlemomenta = []
      times = []
      timesdr = []

      for (i, fileindex) in enumerate(fileindices)
        sim_i = ElectrostaticPIC1D.load(stub, fileindex)

        Ex[:, i] .= sim_i.field.electricfield.(x) ./ electricfieldnormalisation
        Rho[:, i] .= sim_i.field.chargedensity.(x) ./ chargedensitynormalisation

        push!(timesdr, sim_i.time[])

        i % 10 == 0 || continue

        push!(times, sim_i.time[])
        particleenergy = energy(sim_i.plasma)
        fieldenergy = energydensity(sim_i.field) * L
        particlemomentum = momentum(sim_i.plasma)

#        plot!(x, Ex[:, i], label="E")
#        plot!(x, Rho[:, i], label="ρ")
        push!(fieldenergies, fieldenergy)
        push!(particleenergies, particleenergy)
        push!(particlemomenta, particlemomentum)
      end
      @save stub * "energydata.jld2" fieldenergies particleenergies expectedenergy times
      @save stub * "momentumdata.jld2" particlemomenta times
      totalenergies = fieldenergies + particleenergies

      y = fieldenergies / totalenergies[1]
      plot(times, log10.(y), label="field energy")

      y = particleenergies / totalenergies[1]
      plot!(times, log10.(abs.(y)), label="particle energy")

      y = (fieldenergies .+ particleenergies) / expectedenergy
      plot!(times, log10.(abs.(y .- y[1])), label="energy error")

      y = particlemomenta
      plot!(times, log10.(abs.(y .- y[1])), label="momentum")

      xlabel!("Time", legend=:bottomright)
      xlims!(0, maximum(times))
      ylims!(-17, 1.0)
      savefig(stub * "plot.pdf")

      for (Z, fieldstring) ∈ ((Ex, "Ex"), (Rho, "Rho"))
        LogAbsFFTEx = log10.(abs.(fft(Z')));
        LogAbsFFTEx[:, 1] *= NaN; # zeros on the log10 colorbar destroy the contrast
        LogAbsFFTEx[:, (end÷2) + 1 ] *= NaN;
        ks = (-NG÷2 : NG÷2 - 1)
        NT = length(timesdr)
        NT_2 = Int(floor(NT/2))

        ws = (0:NT_2-1) ./ (timesdr[end] - timesdr[1]) / (electronplasmafrequency / 2π)
        Z1 = fftshift(LogAbsFFTEx)[end-length(ws)+1:end, :]
        heatmap(ks, ws, Z1)
        xlabel!("Wavenumber [Waves per box]")
        ylabel!("Frequency [Π]")
        ylims!(0, maximum(ws))
        xlims!(ks[1], ks[end])
        savefig(stub * "$(fieldstring)_dispersionrelation.pdf")
      end
      @save stub * "dispersionrelation.jld2" Ex Rho timesdr x L electronplasmafrequency

    catch e
      @show fieldtypename, particletypename
      @show e
      rethrow(e)
    end

    return nothing
  end

  inputs = Iterators.product(particleshapes, fieldsolvers, particleics)

  map(innerloop, inputs)

end

@testset "Simulations" begin
  go(; datadir="data_maxwellian")
end
