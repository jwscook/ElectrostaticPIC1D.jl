using ElectrostaticPIC1D, Random, Test; Random.seed!(0)

using ForwardDiff, LinearAlgebra

function setup(;N=2^rand(5:7), L=10*rand(), A=10*randn(), p=rand()*2π, n=rand(1:4))
  k = 2π/L * n 
  efield(x) = A * sin(x * k + p)
  rho(x) = k * A * cos(x * k + p)
  return (N, L, efield, rho, A)
end
function findpower(cellsizes, errors)
  p, c = hcat(log10.(cellsizes), ones(length(cellsizes))) \ log10.(errors)
  return p
end

@testset "Fields" begin

numtests = 10

@testset "Fourier fields" begin
  for i in 1:numtests
    N, L, efield, rho, A = setup()
    charge = EquispacedValueGrid(N, L)
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
    charge = EquispacedValueGrid(N, L)
    x = cellcentres(charge)
    efieldexpected = efield.(x)
    rhoexpected = rho.(x)
    charge .= rhoexpected
    for order ∈ keys(data)
      f = FiniteDifferenceField(charge; order=order)
      @test f.charge ≈ rhoexpected atol=10eps()
      solve!(f)
      if i == numtests
        @test f.electricfield ≈ efieldexpected atol=10eps() rtol=0.01^order
      end
      nrm = norm(f.electricfield .- efieldexpected) ./ norm(efieldexpected)
      push!(data[order], (N, nrm))
    end
  end
  for order in keys(data)
    cellsizes = [1 ./i[1] for i in data[order]]
    errors = [i[2] for i in data[order]]
    # fit log10(errors) = p *log10(cellsizes) + c because ϵ ∝ hᵖ
    p = findpower(cellsizes, errors)
    @test (p > order*0.99)
  end
end

@testset "LSFEM fields" begin
  function _dotest(shape; verbose=false)
    nrms = []
    Ns = Int.(exp2.(4:10))
    for N in Ns
      N, L, efield, rho, A = setup(N=N, n=1)
      charge = LSFEMGrid(N, L, shape)
      x = ((1:N) .- 0.5) ./ N * L
      efieldexpected = efield.(x)
      rhoexpected = rho.(x)
      update!(charge, rho)
      f = LSFEMField(charge)
      fieldcharge = f.charge.(x)
      @test fieldcharge ≈ rhoexpected atol=0 rtol=100eps()
      solve!(f)
      efieldresult = f.electricfield.(x)
      nrm = norm(efieldresult .- efieldexpected) ./ norm(efieldexpected)
      verbose && @show N, nrm, shape
      push!(nrms, nrm)
    end
    return nrms, Ns
  end

  @testset "BSpline{1}" begin
    nrms, Ns = _dotest(BSpline{1})
    p = findpower(1 ./ Ns, nrms)
    @test p ≈ 2 rtol = 0.1
  end
  @testset "BSpline{2}" begin
    nrms, Ns = _dotest(BSpline{2})
    inds = Ns .< 1000
    p = findpower(1 ./ Ns[inds], nrms[inds])
    @test p > 2.9
  end
  @testset "Gaussian" begin
    nrms, Ns = _dotest(GaussianShape, verbose=true)
    p = findpower(1 ./ Ns, nrms)
    @test all(nrms .< 0.01)
  end
end

end
