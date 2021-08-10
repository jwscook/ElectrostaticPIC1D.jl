@testset "Fourier fields" begin
  for i in 1:10
    N = 2^rand(5:7)
    L = 10 * rand()
    k = 2π/L * rand(1:N÷2)
    charge = DeltaFunctionGrid(N, L)
    x = cellcentres(charge)
    rho = sin.(x * k)
    efield = -cos.(x * k) ./ k

    charge .= rho
    ff = FourierField(charge::DeltaFunctionGrid{PeriodicGridBC})
    @test ff.charge ≈ rho
    solve!(ff)
    @test ff.electricfield ≈ efield
  end

end
