using NumericalIntegration, Statistics

struct PeriodicFiniteDifferenceOperator{T} <: AbstractMatrix{T}
  N::Int
  order::Int
  accuracy::Int
  stencil::Vector{T}
end
function PeriodicFiniteDifferenceOperator(N::Int, L::T, order::Int, accuracy::Int=2) where {T}
  @assert iseven(accuracy)
  Δ = L / N
  b = zeros(accuracy + 1)
  b[order + 1] = factorial(order)
  x = (collect(1:accuracy + 1) .- (accuracy÷2 + 1)) .* Δ
  stencil = hcat([x[i+1].^(0:accuracy) for i in 0:accuracy]...) \ b ./ Δ^order
  return PeriodicFiniteDifferenceOperator{T}(N, order, accuracy, stencil)
end
Base.eltype(p::PeriodicFiniteDifferenceOperator{T}) where T = T 
Base.size(p::PeriodicFiniteDifferenceOperator{T}) where {T} = (p.N, p.N)
function Base.getindex(p::PeriodicFiniteDifferenceOperator{T},
                       I::Vararg{Int, 2}) where {T}
  i, j = I
  A = p.accuracy
  for jj ∈ (-A÷2:A÷2)
    j == mod1(jj + i, p.N) && return p.stencil[jj + A÷2 + 1]
  end
  return zero(T)
end



struct PeriodicFiniteIntegratorOperator{A,T} <: Function #AbstractMatrix{T}
  N::Int
  Δ::T
  function PeriodicFiniteIntegratorOperator(N::Int, L::T, accuracy::Int=2) where {T}
    Δ = L / N
    1 <= accuracy <=2 || throw(ArgumentError("accuracy must be between 1 & 2"))
    return new{accuracy,T}(N, Δ)
  end
end
Base.eltype(p::PeriodicFiniteIntegratorOperator{A,T}) where {A,T} = T 
Base.size(p::PeriodicFiniteIntegratorOperator) = (p.N, p.N)
#function Base.getindex(p::PeriodicFiniteIntegratorOperator{1,T},
#                       I::Vararg{Int, 2}) where {T}
#  i, j = I
#  return j > i ? zero(T) : p.Δ
#end
#function Base.getindex(p::PeriodicFiniteIntegratorOperator{2,T},
#                       I::Vararg{Int, 2}) where {T}
#  i, j = I
#  i == 1 && (j == 1 || j == p.N) && return p.Δ / 2
#  return j > i ? zero(T) : p.Δ
#end
demean!(x) = (x .-= mean(x); x)
function (p::PeriodicFiniteIntegratorOperator{1})(z, y)
  z .= cumsum(y) * p.Δ
  return demean!(z) 
end
function (p::PeriodicFiniteIntegratorOperator{2})(z, y)
#  z .= cumul_integrate(p.Δ/2:p.Δ:p.N*p.Δ, y, TrapezoidalEven())
  z[1] = (y[end]/4 + y[1]/2 + y[2]/4) * p.Δ
  for i in 2:length(z)-1
    z[i] = z[i-1] + (y[i-1]/4 + y[i]/2 + y[i+1]/4) * p.Δ
  end
  z[end] = z[end-1] + (y[end]/4 + y[1]/2 + y[2]/4) * p.Δ
  return demean!(z) 
end
function (p::PeriodicFiniteIntegratorOperator{3})(z, y)
  z .= cumul_integrate(p.Δ/2:p.Δ:p.N*p.Δ, y, SimpsonEven())
  return demean!(z) 
end
(p::PeriodicFiniteIntegratorOperator{A,T})(y) where {A,T} = p(deepcopy(y), y)


struct FiniteDifferenceField{BC<:PeriodicGridBC, T, A} <: AbstractField{BC}
  charge::DeltaFunctionGrid{BC,T}
  electricfield::DeltaFunctionGrid{BC,T}
#  gradient::PeriodicFiniteDifferenceOperator{T}
  #laplace::PeriodicFiniteIntegratorOperator{T}
  integrator::PeriodicFiniteIntegratorOperator{A,T}
end
function FiniteDifferenceField(charge::DeltaFunctionGrid{PeriodicGridBC},
                               accuracy::Int=2)
  electricfield = deepcopy(charge)
  zero!(electricfield)
  N = first(size(charge))
#  gradient = PeriodicFiniteDifferenceOperator(N, charge.L, 1, accuracy)
  #laplace = PeriodicFiniteIntegratorOperator(N, charge.L, 2, accuracy)
  integrator = PeriodicFiniteIntegratorOperator(N, charge.L, accuracy)
  return FiniteDifferenceField(charge, electricfield, integrator)
end

function solve!(f::FiniteDifferenceField)
  f.electricfield .= f.integrator(f.charge)
  return f
end

function update!(f::FiniteDifferenceField, species)
  for particle ∈ species
    deposit!(f.charge, particle)
  end
  return f
end
