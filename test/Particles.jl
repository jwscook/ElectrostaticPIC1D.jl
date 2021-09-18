using ElectrostaticPIC1D, Random, Statistics, Test; Random.seed!(0)


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

  @testset "Deposition" begin
    @testset "DeltaFunctionShape, EquispacedValueGrid" begin
      shape = DeltaFunctionShape()
      for _ ∈ 1:1
        N, L, w, q = Int(exp2(rand(3:8))), 10.0 * rand(), rand(), rand()
        p = Particle(Nuclide(q, 1.0), shape;
                     position=rand()*L, velocity=0.0, weight=w)
        @test weight(p) == w
        @test charge(p) == q
        rho = EquispacedValueGrid(N, L)
        f = FourierField(rho)
        deposit!(f, p)
        particleindex = cell(centre(p), rho)
        @test f.charge[particleindex] ≈ w * q rtol=eps()
        deposit!(f, p)
        @test f.charge[particleindex] ≈ 2w * q rtol=eps()
        E = rand()
        f.electricfield .= E
        @test antideposit(f, p) ≈ E rtol=eps()
      end
    end
    @testset "DeltaFunctionShape, LSFEM TopHatShapes" begin
      shape = DeltaFunctionShape()
      for _ ∈ 1:1
        N, L, w, q = Int(exp2(rand(3:8))), 10.0 * rand(), 3*rand(), rand()
        p = Particle(Nuclide(q, 1.0), shape;
                     position=rand()*L, velocity=0.0, weight=w)
        @test weight(p) == w
        @test charge(p) == q
        rho = FEMGrid(N, L, TopHatShape)
        f = LSFEMField(rho)
        deposit!(f, p)
        for i in f.charge
          if in(i, basis(p))
            @test weight(i) ≈ w * q rtol=eps()
          else
            @test weight(i) == 0.0
          end
        end
        E = rand()
        f.electricfield .= E
        @test antideposit(f, p) ≈ E rtol=eps()
      end
    end
  end

  @testset "Lots of particles tests" begin
    NG = 32
    NP = NG * 4
    L = 1.0# rand()
    Δ = L / NG
    nuclide = Nuclide(rand(2)...)

    physicaldensity = NP / L#make weight unity #rand()
    weight = physicaldensity * L / NP

    particleshapes = ((DeltaFunctionShape(), "Delta"),
                      (GaussianShape(Δ), "Gaussian"),
                      (BSpline{0}(Δ), "BSpline0"),
                      (BSpline{1}(Δ), "BSpline1"),
                      (BSpline{2}(Δ), "BSpline2"),)

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

    xs = collect(0:1/NP:1-1/NP) .* L # equi-spaced

    for (particleshape, pname) ∈ particleshapes
      species = Species([Particle(nuclide, particleshape; x=x, v=0.0, w=weight) for x in xs])

      expectedchargedensity = weight * NP * charge(nuclide)
      plasma = Plasma([species])

      @test expectedchargedensity ≈ chargedensity(plasma)

      for (field, fname) ∈  fieldsolvers
        @testset  "$fname-$pname" begin
          ElectrostaticPIC1D.zero!(field)
          deposit!(field, plasma)
          qs = field.charge.(xs)
          @test sqrt(mean((qs .- mean(qs)).^2)) ./ mean(qs) < 1e-6
          r = chargedensity(field) / expectedchargedensity
          @test chargedensity(field) ≈ expectedchargedensity
        end
      end
    end

  end


end
