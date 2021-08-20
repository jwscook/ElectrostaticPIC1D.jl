using Statistics, Memoization

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
    return new{accuracy,T}(N, Δ)
  end
end
Base.eltype(p::PeriodicFiniteIntegratorOperator{A,T}) where {A,T} = T 
Base.size(p::PeriodicFiniteIntegratorOperator) = (p.N, p.N)
@memoize polymatrix(N) = lu(hcat([(0:N-1).^i for i in 0:N-1]...))
function integratepoly(y, a, b) # TODO calculate coefficients and not do this
  N = length(y)
  N > 8 && throw(error("length of y is too large, gives badly conditioned matrix"))
  coeffs = zeros(N+1)
  coeffs[2:end] = (polymatrix(N) \ y) ./ (1:N)
  return sum(coeffs .* b.^(0:N)) - sum(coeffs .* a.^(0:N))
end
function (p::PeriodicFiniteIntegratorOperator{1})(z, y)
  z[1] = y[1]/2 * p.Δ
  for i ∈ 2:length(z)
    z[i] = z[i-1] + y[i] * p.Δ
  end
  return demean!(z) 
end
function (p::PeriodicFiniteIntegratorOperator{2})(z, y)
  z[1] = (y[end]/4 + 3y[1]/4) * p.Δ
  for i in 2:length(z)
    z[i] = z[i-1] + (y[i-1] + y[i])/2 * p.Δ
  end
  return demean!(z) 
end
# this operator is 4th order accurate despite using a 3 point stencil
function (p::PeriodicFiniteIntegratorOperator{3})(z, y)
  fill!(z, 0)
  z[1] = integratepoly([y[end], y[1], y[2]], 0.5, 1)
  z[2] = integratepoly([y[end], y[1], y[2]], 1, 1.5)
  for i in 2:length(z)-1
    z[i] += z[i-1] + integratepoly(y[i-1:i+1], 0.5, 1)
    z[i+1] += integratepoly(y[i-1:i+1], 1, 1.5)
  end
  z[end] += z[end-1] + integratepoly([y[end-1], y[end], y[1]], 0.5, 1)
  return demean!(z) * p.Δ
end
function (p::PeriodicFiniteIntegratorOperator{4})(z, y)
  z[1] = integratepoly([y[end-1], y[end], y[1], y[2]], 1.5, 2)
  z[2] = z[1] + integratepoly([y[end], y[1], y[2], y[3]], 1, 2)
  for i in 3:length(z)-1
    z[i] = z[i-1] + integratepoly(y[i-2:i+1], 1, 2)
  end
  z[end] = z[end-1] + integratepoly([y[end-2], y[end-1], y[end], y[1]], 1, 2)
  return demean!(z) * p.Δ
end
(p::PeriodicFiniteIntegratorOperator{A,T})(y) where {A,T} = p(deepcopy(y), y)


struct FiniteDifferenceField{BC<:PeriodicGridBC, T, A} <: AbstractField{BC}
  charge::EquispacedValueGrid{BC,T}
  electricfield::EquispacedValueGrid{BC,T}
  integrator::PeriodicFiniteIntegratorOperator{A,T}
end
function FiniteDifferenceField(charge::EquispacedValueGrid{PeriodicGridBC};
                               accuracy::Int=2)
  electricfield = deepcopy(charge)
  zero!(electricfield)
  N = first(size(charge))
  integrator = PeriodicFiniteIntegratorOperator(N, charge.L, accuracy)
  return FiniteDifferenceField(charge, electricfield, integrator)
end

accuracy(f::FiniteDifferenceField{BC,T,A}) where {BC,T,A} = A

function solve!(f::FiniteDifferenceField)
  f.electricfield .= f.integrator(f.charge)
  return f
end

