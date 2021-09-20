using ElectrostaticPIC1D, QuadGK, Random, Statistics, Test; Random.seed!(0)


@testset "Particles" begin
  @testset "Basic" begin
    q, m = 1.6, 9.11
    nuclide = Nuclide(q, m)
    pos = 1.0
    vel = 2.0
    particle = Particle(nuclide; position=pos, velocity=vel, weight=0.0)
    dt = 3.0
    pushposition!(particle, dt, PeriodicGridBC(-1e16, 1e16))
    @test position(particle) .== pos + vel * dt
    efield = 4.0
    pushvelocity!(particle, efield, dt)
    @test velocity(particle) .== vel + q / m * efield * dt
  end

  @testset "Periodic BCs" begin
    nuclide = Nuclide(1.0, 1.0)
    pos = 0.5
    vel = 0.3
    particle = Particle(nuclide; position=pos, velocity=vel, weight=0.0)
    dt = 1.0
    pushposition!(particle, dt, PeriodicGridBC(0.0, 1.0))
    @test position(particle) .== mod(pos + vel * dt, 1.0)
    pushposition!(particle, dt, PeriodicGridBC(0.0, 1.0))
    @test position(particle) .== mod(pos + 2vel * dt, 1.0)
  end

  @testset "Particles basis functions are normalised" begin
    nuclide = Nuclide(rand(2)...)
    Δ = rand()
    particleshapes = ((BSpline{0}(Δ), "BSpline0"),
                      (BSpline{1}(Δ), "BSpline1"),
                      (BSpline{2}(Δ), "BSpline2"),
                      (DeltaFunctionShape(), "Delta"),
                      (GaussianShape(Δ), "Gaussian"),)
    for (particleshape, pname) ∈ particleshapes
      part = Particle(nuclide, particleshape; x=rand(), v=rand(), w=rand())
      b = ElectrostaticPIC1D.basis(part)
      area = ElectrostaticPIC1D.integral(b, x->1,
        ElectrostaticPIC1D.PeriodicGridBC(1.0))
      @test area ≈ 1
    end
  end

  @testset "Lots of particles tests" begin
    NG = 128 # number of grid points / basis functions
    NPPC = 2 # number of particle per cell
    NP = NG * NPPC
    L = 1.0 # rand()
    Δ = L / NG
    nuclide = Nuclide(rand(2)...)

    physicaldensity = NP / L#make weight unity #rand()
    weight0 = physicaldensity * L / NP

    particleshapes = ((BSpline{0}(Δ), "BSpline0"),
                      (BSpline{1}(Δ), "BSpline1"),
                      (BSpline{2}(Δ), "BSpline2"),
                      (DeltaFunctionShape(), "Delta"),
                      (GaussianShape(Δ), "Gaussian"),)

    fieldsolvers = (
      (FourierField(NG,L), "Fourier"),
      (LSFEMField(NG,L,BSpline{1}(Δ)), "LSFEM_BSpline1"),
      (LSFEMField(NG,L,BSpline{2}(Δ)), "LSFEM_BSpline2"),
      (LSFEMField(NG,L,GaussianShape(Δ * √2)), "LSFEM_Gaussian"),
      (GalerkinFEMField(NG,L,BSpline{0}(Δ), BSpline{1}(Δ)), "Galerkin_BSpline0_BSpline1"),
      (GalerkinFEMField(NG,L,BSpline{1}(Δ), BSpline{2}(Δ)), "Galerkin_BSpline1_BSpline2"),
      (FiniteDifferenceField(NG,L,order=1), "FiniteDifference1"),
      (FiniteDifferenceField(NG,L,order=2), "FiniteDifference2"),
      (FiniteDifferenceField(NG,L,order=4), "FiniteDifference4"),
      )
    q = charge(nuclide)
    xs = collect(0:1/NP:1-1/NP) .* L # equi-spaced

    perturbations = (0.0, 1e-8)

    for (particleshape, pname) ∈ particleshapes, pert ∈ perturbations
      pertname = iszero(pert) ? "Uniform" : "Wave"
      ρ0 = q * physicaldensity
      chargedensityfun(x) = ρ0 * (1.0 + pert * sin(2π*x/L + π/4))
      weightfun(x) = chargedensityfun(x) * L / NP / q
      E0 = q * physicaldensity * pert * L/2π
      expectedelectricfield(x) = - E0 * cos(2π*x/L + π/4)

      species = Species([Particle(nuclide, particleshape; x=x, v=0.0,
        w=weightfun(x)) for x in xs])

      expectedtotalchargedensity = weight0 * NP * q
      totalweight = sum(ElectrostaticPIC1D.weight.(species))
      @test totalweight * q ≈ expectedtotalchargedensity * L
      plasma = Plasma([species])

      @test expectedtotalchargedensity ≈ chargedensity(plasma)

      for (field, fname) ∈  fieldsolvers
        @testset  "$pname-$pertname-$fname" begin
          ElectrostaticPIC1D.zero!(field)
          try
            @inferred deposit!(field, plasma)
            @test true
          catch
            ElectrostaticPIC1D.zero!(field)
            deposit!(field, plasma)
            @test false
          end
          qs = field.chargedensity.(xs)
          r = chargedensity(field) / expectedtotalchargedensity
          @test chargedensity(field) ≈ expectedtotalchargedensity
          if pert == 0.0
            @test sqrt(mean((qs .- mean(qs)).^2)) ./ mean(qs) < 1e-6
          else
            solve!(field)
            for particle ∈ species
              answer = electricfield(field, particle)
              approxexpected = expectedelectricfield(centre(particle))
              @test approxexpected ≈ answer rtol=0.1 atol=E0/1000
              answer = chargedensity(field, particle)
              approxexpected = chargedensityfun(centre(particle))
              @test approxexpected ≈ answer rtol=0.1 atol=ρ0/1000
            end
          end
        end
      end
    end

  end


end
