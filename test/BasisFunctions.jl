using ElectrostaticPIC1D, Random, Test; Random.seed!(0)

using ForwardDiff, QuadGK

@testset "BasisFunctions" begin

numtests = 10

function isnormalisedtest(Shape)
  outcome = true
  for _ ∈ 1:numtests
    w = rand()
    s = Shape(w)
    x0 = randn()
    integral = quadgk(x->s(x, x0), x0 - 10*w, x0 + 10*w, order=27, rtol=eps())[1]
    outcome = integral ≈ 1
    outcome || break
  end
  @test outcome
end
function supporttest(Shape)
  for _ ∈ 1:10
    w = rand()
    x0 = randn()
    b = BasisFunction(Shape(w), x0)
    outcome = true
    for _ in 1:1000
      x = 3 * randn() * w + x0
      if x ∈ b
        outcome = b(x) > 0
        outcome || @show b, x, b(x)
      else
        outcome = b(x) <= b(x0) * eps()
        outcome || @show b, x, b(x)
      end
      outcome || break
    end
    @test outcome
  end
end
function energytest(Shape)
  outcome = true
  for _ ∈ 1:numtests
    w = rand()
    x0 = randn()
    b = BasisFunction(Shape(w), x0)
    expected = quadgk(x->b(x)^2, x0 - 10*w, x0 + 10*w, order=27, rtol=eps())[1] * weight(b)^2 / 2
    result = integral(b, b, PeriodicGridBC(x0-10*w, x0+10*w)) * weight(b)^2 / 2
    outcome = expected ≈ result
    outcome || break
  end
  @test outcome
end

for S ∈ (BSpline{0}, BSpline{1}, BSpline{2}, GaussianShape)
 @testset "$S" begin
   isnormalisedtest(S)
   supporttest(S)
   energytest(S)
 end
end

@testset "Pairwise Integrals" begin
for U ∈ (DeltaFunctionShape, BSpline{0}, BSpline{1}, BSpline{2}, GaussianShape)
for V ∈ (BSpline{0}, BSpline{1}, BSpline{2}, GaussianShape)
@testset "$U $V" begin
  noverlaps = 0
  while true
    v = BasisFunction(V(5*rand()), randn())
    u = if U <: DeltaFunctionShape
      BasisFunction(U(), lower(v) + rand() * width(v))
    else
      BasisFunction(U(5*rand()), randn())
    end
    expected = if U <: DeltaFunctionShape
      v(centre(u))
    else
      quadgk(x-> u(x) * v(x), lower(u, v), upper(u, v);
             order=51, atol=0, rtol=eps())[1]
    end
    expected < eps() && continue
    try
      @inferred integral(u, v)
      @test true
    catch
      @test false
    end
    try
      @inferred integral(u, v, PeriodicGridBC(1.0))
      @test true
    catch
      @warn "integral($U, $V) cannot be inferred"
      @test false
    end
    result = integral(u, v)
    @test result ≈ expected atol=100eps()
    break
  end
end # testset U, V
end # V
end # U
end # testset integrals
 
end
