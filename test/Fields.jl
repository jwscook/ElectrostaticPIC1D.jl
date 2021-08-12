using ForwardDiff, LinearAlgebra

function setup(;N=2^rand(5:7), L=10*rand(), A=10*randn(), p=rand()*2π, n=rand(1:4))
  k = 2π/L * n 
  efield(x) = A * sin(x * k + p)
  rho(x) = k * A * cos(x * k + p)
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
  data = Dict(1=>[], 2=>[], 3=>[], 4=>[])
  for i in 1:8
    N, L, efield, rho, A = setup(N=2^(i+3), n=1, A=1)
    charge = DeltaFunctionGrid(N, L)
    x = cellcentres(charge)
    efieldexpected = efield.(x)
    rhoexpected = rho.(x)
    charge .= rhoexpected
    for accuracy ∈ keys(data)
      f = FiniteDifferenceField(charge; accuracy=accuracy)
      @test f.charge ≈ rhoexpected atol=10eps()
      solve!(f)
      if i == numtests
        @test f.electricfield ≈ efieldexpected atol=10eps() rtol=0.01^accuracy
      end
      nrm = norm(f.electricfield .- efieldexpected) ./ norm(efieldexpected)
      push!(data[accuracy], (N, nrm))
    end
  end
  for accuracy in keys(data)
    cellsizes = [1 ./i[1] for i in data[accuracy]]
    errors = [i[2] for i in data[accuracy]]
    # fit log10(errors) = p *log10(cellsizes) + c because ϵ ∝ hᵖ
    p, c = hcat(log10.(cellsizes), ones(length(cellsizes))) \ log10.(errors)
    @test (p > accuracy*0.99)
  end
end

@testset "LSFEM fields" begin
  for i in 1:numtests
    N, L, efield, rho, A = setup()
    charge = LSFEMGrid(N, L, GaussianShape)
    x = ((1:N) .- 0.5) ./ N * L
    efieldexpected = efield.(x)
    rhoexpected = rho.(x)
    update!(charge, rho)
    f = LSFEMField(charge)
    fieldcharge = f.charge.(x)
    @test fieldcharge ≈ rhoexpected atol=10eps()
  #  solve!(f)
  #  fieldelectricfield = f.electricfield.(x)
  #  @test fieldelectricfield ≈ efieldexpected atol=10eps() rtol=sqrt(eps())
  end
end

end
