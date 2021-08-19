using ElectrostaticPIC1D, Random, Test

Random.seed!(0)

@testset "ElectrostaticPIC1D tests" begin
  include("./BasisFunctions.jl")
  include("./Fields.jl")
  include("./Particles.jl")
end
