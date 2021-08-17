using ElectrostaticPIC1D, Random, Test; Random.seed!(0)


@testset "Particles" begin
  @testset "Basic" begin
    q, m = 1.6, 9.11
    nuclide = Nuclide(q, m)
    basis = BasisFunction(DeltaFunctionShape(), 0.0)
    pos = 1.0
    vel = 2.0
    particle = Particle(nuclide, basis, pos, vel)
    dt = 3.0
    pushposition!(particle, dt)
    @test position(particle) .== pos + vel * dt
    efield = 4.0
    pushvelocity!(particle, efield, dt)
    @test velocity(particle) .== vel + q / m * efield * dt
  end
  @testset "deposition" begin
    N, L = 8, 8.0
    basis = BasisFunction(DeltaFunctionShape(), 0.0)
    weight = Float64(π)
    p = Particle(Nuclide(1.0, 1.0), basis, L/4, weight)

    charge = DeltaFunctionGrid(N, L)
    @show union(p, charge)
    f = FourierField(charge)
    @show f.charge
    deposit!(f, p)
    @show f.charge
  end
end
