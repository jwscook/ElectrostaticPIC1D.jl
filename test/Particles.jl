using ElectrostaticPIC1D, Random, Test; Random.seed!(0)


@testset "Particles" begin
  electron = Nuclide(1.6e-19, 9.11e-31)
  basis = BasisFunction(DeltaFunctionShape(), 0.0)

  particle = Particle(electron, basis)

end
