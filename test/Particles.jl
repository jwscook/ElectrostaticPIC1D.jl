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

  @testset "deposition" begin
    N, L = 8, 8.0
    shape = DeltaFunctionShape()
    p = Particle(Nuclide(1.0, 1.0), shape;
                 position=L/4, velocity=0.0, weight=1.0)
    @test weight(p) == 1.0

    rho = DeltaFunctionGrid(N, L)
    f = FourierField(rho)
    deposit!(f, p)
    @show f.charge
  end


end
