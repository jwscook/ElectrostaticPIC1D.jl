using ForwardDiff

function setup()
  N = 2^rand(5:7)
  L = 10 * rand()
  k = 2π/L * rand(1:N÷4)
  A = 10 * randn()
  efield(x) = A * sin(x * k)
  rho(x) = k * A * cos(x * k)
  return (N, L, efield, rho, A)
end
@testset "Fields" begin

numtests = 10

@testset "Fourier fields" begin
  for i in 1:numtests
    N, L, efield, rho, A = setup()
    charge = DeltaFunctionGrid(N, L)
    x = cellcentres(charge)
    efieldexpected = efield.(x)
    rhoexpected = rho.(x)
    charge .= rhoexpected
    f = FourierField(charge)
    @test f.charge ≈ rhoexpected atol=10eps()
    solve!(f)
    @test f.electricfield ≈ efieldexpected atol=10eps() rtol=sqrt(eps())
  end
end

@testset "FiniteDifference fields" begin
  for i in 1:numtests
    N, L, efield, rho, A = setup()
    accuracy = rand(2:2:N÷4)
    charge = DeltaFunctionGrid(N, L)
    x = cellcentres(charge)
    efieldexpected = efield.(x)
    rhoexpected = rho.(x)
    charge .= rhoexpected
    f = FiniteDifferenceField(charge)
    @test f.charge ≈ rhoexpected atol=10eps()
    solve!(f)
    @test f.electricfield ≈ efieldexpected atol=10eps() rtol=sqrt(eps())
  end

end

end
