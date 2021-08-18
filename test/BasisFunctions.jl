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
    outcome = outcome && integral ≈ 1
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
        outcome = outcome && b(x) > 0
        outcome || @show b, x, b(x)
      else
        outcome = outcome && b(x) <= b(x0) * eps()
        outcome || @show b, x, b(x)
      end
      outcome || break
    end
    @test outcome
  end
end

@testset "BSpline{0}" begin
  isnormalisedtest(BSpline{0})
  supporttest(BSpline{0})
end
@testset "BSpline{1}" begin
  isnormalisedtest(BSpline{1})
  supporttest(BSpline{1})
end
@testset "BSpline{2}" begin
  isnormalisedtest(BSpline{2})
  supporttest(BSpline{2})
end
@testset "GaussianShape" begin
  isnormalisedtest(GaussianShape)
  supporttest(GaussianShape)
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
    result = integral(u, v)
    @test result ≈ expected atol=100eps()
    break
  end
end # testset U, V
end # V
end # U
end # testset integrals
 
end
