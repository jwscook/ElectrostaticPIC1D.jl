using ElectrostaticPIC1D, Random, Test; Random.seed!(0)


@testset "Particles" begin
  @testset "Basic" begin
    q, m = 1.6, 9.11
    nuclide = Nuclide(q, m)
    pos = 1.0
    vel = 2.0
    particle = Particle(nuclide; position=pos, velocity=vel, weight=0.0)
    dt = 3.0
    pushposition!(particle, dt)
    @test position(particle) .== pos + vel * dt
    efield = 4.0
    pushvelocity!(particle, efield, dt)
    @test velocity(particle) .== vel + q / m * efield * dt
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
        @show E
        @show f.electricfield.values
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
        rho = LSFEMGrid(N, L, TopHatShape)
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
        @show E
        @show weight.(f.electricfield.bases)
        @test antideposit(f, p) ≈ E rtol=eps()
      end
    end
  end


end
