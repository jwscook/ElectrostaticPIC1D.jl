using Statistics, Memoization

struct PeriodicFiniteDifferenceOperator{T} <: AbstractMatrix{T}
  N::Int
  difforder::Int
  accuracyorder::Int
  stencil::Vector{T}
end
function PeriodicFiniteDifferenceOperator(N::Int, L::T, difforder::Int, accuracyorder::Int=2) where {T}
  @assert iseven(accuracyorder)
  Δ = L / N
  b = zeros(accuracyorder + 1)
  b[difforder + 1] = factorial(difforder)
  x = (collect(1:accuracyorder + 1) .- (accuracy÷2 + 1)) .* Δ
  stencil = hcat([x[i+1].^(0:accuracyorder) for i in 0:accuracyorder]...) \ b ./ Δ^difforder
  return PeriodicFiniteDifferenceOperator{T}(N, difforder, accuracyorder, stencil)
end
Base.eltype(p::PeriodicFiniteDifferenceOperator{T}) where T = T 
Base.size(p::PeriodicFiniteDifferenceOperator{T}) where {T} = (p.N, p.N)
function Base.getindex(p::PeriodicFiniteDifferenceOperator{T},
                       I::Vararg{Int, 2}) where {T}
  i, j = I
  A = p.order
  for jj ∈ (-A÷2:A÷2)
    j == mod1(jj + i, p.N) && return p.stencil[jj + A÷2 + 1]
  end
  return zero(T)
end



struct PeriodicFiniteIntegratorOperator{A,T} <: Function
  N::Int
  Δ::T
  function PeriodicFiniteIntegratorOperator(N::Int, L::T, order::Int=2) where {T}
    Δ = L / N
    return new{order,T}(N, Δ)
  end
end
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
  return z
end
function (p::PeriodicFiniteIntegratorOperator{2})(z, y)
  z[1] = (y[end]/4 + 3y[1]/4) * p.Δ
  for i in 2:length(z)
    z[i] = z[i-1] + (y[i-1] + y[i])/2 * p.Δ
  end
  return z
end
# this operator is 4th order accurate despite using a 3 point stencil
function (p::PeriodicFiniteIntegratorOperator{3})(z, y)
  fill!(z, 0)
  z[1] = integratepoly([y[end], y[1], y[2]], 0.5, 1) * p.Δ
  z[2] = integratepoly([y[end], y[1], y[2]], 1, 1.5) * p.Δ
  for i in 2:length(z)-1
    z[i] += z[i-1] + integratepoly(y[i-1:i+1], 0.5, 1) * p.Δ
    z[i+1] += integratepoly(y[i-1:i+1], 1, 1.5) * p.Δ
  end
  z[end] += z[end-1] + integratepoly([y[end-1], y[end], y[1]], 0.5, 1) * p.Δ
  return z
end
function (p::PeriodicFiniteIntegratorOperator{4})(z, y)
  z[1] = integratepoly([y[end-1], y[end], y[1], y[2]], 1.5, 2) * p.Δ
  z[2] = z[1] + integratepoly([y[end], y[1], y[2], y[3]], 1, 2) * p.Δ
  for i in 3:length(z)-1
    z[i] = z[i-1] + integratepoly(y[i-2:i+1], 1, 2) * p.Δ
  end
  z[end] = z[end-1] + integratepoly([y[end-2], y[end-1], y[end], y[1]], 1, 2) * p.Δ
  return z
end
(p::PeriodicFiniteIntegratorOperator{A,T})(y) where {A,T} = p(deepcopy(y), y)


struct FiniteDifferenceField{BC<:PeriodicGridBC, T, A} <: AbstractField{BC}
  chargedensity::EquispacedValueGrid{BC,T}
  electricfield::EquispacedValueGrid{BC,T}
  integrator::PeriodicFiniteIntegratorOperator{A,T}
  function FiniteDifferenceField(chargedensity::EquispacedValueGrid{BC,T},
                                 electricfield::EquispacedValueGrid{BC,T},
                                 integrator::PeriodicFiniteIntegratorOperator{A,T}) where {BC,T,A}
    finitediff = new{BC,T,A}(chargedensity, electricfield, integrator)
    zero!(finitediff)
    return finitediff
  end
end
function FiniteDifferenceField(chargedensity::EquispacedValueGrid{PeriodicGridBC};
                               order::Int=2)
  electricfield = deepcopy(chargedensity)
  zero!(electricfield)
  N = first(size(chargedensity))
  integrator = PeriodicFiniteIntegratorOperator(N, chargedensity.L, order)
  return FiniteDifferenceField(chargedensity, electricfield, integrator)
end
function FiniteDifferenceField(N::Int, L::Real, ::Type{BC}=PeriodicGridBC; order::Int=2) where {BC}
  return FiniteDifferenceField(EquispacedValueGrid(N, L, BC), order=order)
end

presolve(g::EquispacedValueGrid{PeriodicGridBC}) = g .- mean(g)
presolve(g::EquispacedValueGrid{<:AbstractBC}) = g
function solve!(f::FiniteDifferenceField{PeriodicGridBC})
  ρ = presolve(f.chargedensity)
  f.electricfield .= demean!(f.integrator(ρ))
  return f
end

zero!(f::FiniteDifferenceField) = (zero!(f.chargedensity); zero!(f.electricfield))

