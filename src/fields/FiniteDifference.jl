struct PeriodicFiniteDifferenceOperator{T} <: AbstractMatrix{T}
  N::Int
  order::Int
  accuracy::Int
  stencil::Vector{T}
end
function PeriodicFiniteDifferenceOperator(N::Int, L::T, order::Int, accuracy::Int=2) where {T}
  @assert iseven(accuracy)
  Δ = L / N
  b = zeros(order + 1)
  b[order + 1] = factorial(order)
  x = (collect(1:accuracy + 1) .- (accuracy÷2 + 1)) .* Δ
  stencil = hcat([x.^p for p in 0:accuracy]...)' \ b ./ Δ^order
  return PeriodicFiniteDifferenceOperator{T}(N, order, accuracy, stencil)
end
Base.eltype(p::PeriodicFiniteDifferenceOperator{T}) where T = T 
Base.size(p::PeriodicFiniteDifferenceOperator{T}) where {T} = (p.N, p.N)
function Base.getindex(p::PeriodicFiniteDifferenceOperator{T}, i, j
    ) where {T}
  A = p.accuracy
  for jj ∈ (-A÷2:A÷2)
    j == mod1(jj + i, p.N) && return p.laplacestencil[jj + A÷2 + 1]
  end
  return zero(T)
end

struct FiniteDifferenceField{BC<:PeriodicGridBC, T} <: AbstractField{BC}
  charge::DeltaFunctionGrid{BC,T}
  electricfield::DeltaFunctionGrid{BC,T}
  gradient::PeriodicFiniteDifferenceOperator{T}
  laplace::PeriodicFiniteDifferenceOperator{T}
end
function FiniteDifferenceField(charge::DeltaFunctionGrid{PeriodicGridBC},
                               accuracy::Int=2)
  electricfield = deepcopy(charge)
  zero!(electricfield)
  gradient = PeriodicFiniteDifferenceOperator(size(charge), charge.L, 1, accuracy)
  laplace = PeriodicFiniteDifferenceOperator(size(charge), charge.L, 2, accuracy)
  return FourierField(charge, electricfield, gradient, laplace)
end

function solve!(f::FiniteDifferenceField)
  f.electricfield .= f.gradient * (f.charge \ f.laplace)
  return f
end

function update!(f::FiniteDifferenceField, species)
  for particle ∈ species
    deposit!(f.charge, particle)
  end
  return f
end
