using Memoize, SpecialFunctions

abstract type AbstractShape end
struct GaussianShape <: AbstractShape
  σ::Float64
end
width(s::GaussianShape) = (-6, 6) .* s.σ
(s::GaussianShape)(x, centre) = exp(-(x-centre)^2 / s.σ^2) / √π / s.σ

struct TopHat <: AbstractShape
  fullwidth::Float64
end
width(s::HopHat) = s.fullwidth
(s::TopHat)(x, centre) = centre - s.fullwidth/2 <= x < centre + s.fullwidth/2

struct DeltaFunction <: AbstractShape end
width(s::DeltaFunction) = 0
(s::DeltaFunction)(x, centre) = x == centre

struct BasisFunction{S<:AbstractShape}
  shape::S
  centre::Float64
  weight::Float64
end
lower(b::BasisFunction) = b.centre - width(b.shape) / 2
upper(b::BasisFunction) = b.centre + width(b.shape) / 2
lower(a::T, b::T) where {T<:BasisFunction} = max(lower(a), lower(b))
upper(a::T, b::T) where {T<:BasisFunction} = min(upper(a), upper(b))

(b::BasisFunction)(x) = b.shape(x)

function overlap(a::BasisFunction, b::BasisFunction)
  return lower(a) < upper(b) && upper(a) > lower(b)
end

function BasisFunction(p::Particle{S}) where {S<:AbstractShape}
  return BasisFunction{S}(shape(p), p.x, 1.0)
end
BasisFunction(c::Cell) = BasisFunction(TopHat(width(c)), centre(c))

function integral(p::BasisFunction, limits)
  lower, upper = limits
  @assert lower < upper
  return integral(p, lower, upper)
end

function integral(a::BasisFunction{GaussianShape}, b::BasisFunction{TopHat})
  return (erf(-(a.centre - upper(b)) / a.shape.σ) -
          erf(-(a.centre - lower(b)) / a.shape.σ))/2
end
function integral(a::BasisFunction{TopHat}, b::BasisFunction{TopHat})
  return (min(upper(a), upper(b)) - max(lower(a), lower(b))) * overlap(a, b)
end
integral(a::BasisFunction{DeltaFunction}, b::BasisFunction) = b(a.centre)
integral(a::BasisFunction, b::BasisFunction{DeltaFunction}) = a(b.centre)

function integral(a::BasisFunction{GaussianShape},
                  b::BasisFunction{GaussianShape})
  σ₁, σ₂ = a.shape.σ, b.shape.σ
  μ₁, μ₂ = a.centre, b.centre
  σ² = σ₁^2 + σ₂^2
  return √π * σ₁ * σ₂ / σ² * exp(-(μ₁ - μ₂)^2 / σ²)
end

