using Memoization, SpecialFunctions, Setfield

abstract type AbstractShape end
struct GaussianShape <: AbstractShape
  σ::Float64
end
width(s::GaussianShape) = 12 .* s.σ
(s::GaussianShape)(x, centre) = exp(-(x-centre)^2 / s.σ^2) / √π / s.σ

struct TopHat <: AbstractShape
  fullwidth::Float64
end
width(s::TopHat) = s.fullwidth
function (s::TopHat)(x, centre)
  return (centre - s.fullwidth/2 <= x < centre + s.fullwidth/2) / s.fullwidth
end

struct DeltaFunction <: AbstractShape end
DeltaFunction(_) = DeltaFunction() # so has same ctor as others
width(s::DeltaFunction) = 0
(s::DeltaFunction)(x, centre) = x == centre

mutable struct BasisFunction{S<:AbstractShape, T}
  shape::S
  centre::Float64
  weight::T
end
BasisFunction(s::AbstractShape, centre) = BasisFunction(s, centre, 0.0)
lower(b::BasisFunction) = b.centre - width(b.shape) / 2
upper(b::BasisFunction) = b.centre + width(b.shape) / 2
lower(a::T, b::T) where {T<:BasisFunction} = max(lower(a), lower(b))
upper(a::T, b::T) where {T<:BasisFunction} = min(upper(a), upper(b))
width(b::BasisFunction) = width(b.shape)
weight(b::BasisFunction) = b.weight
centre(b::BasisFunction) = b.centre
function translate(b::BasisFunction, x)
  translated = deepcopy(b)
  translated.centre += x
  return translated
end
Base.:+(b::BasisFunction, x) = (b.weight += x; b)
Base.:*(x, b::BasisFunction) = x * b.weight
Base.:*(b::BasisFunction, x) = x * b.weight
zero!(b::BasisFunction) = (b.weight *= false)
Base.in(x, b::BasisFunction) = (b.centre - width(b)/2 <= x < b.centre + width(b)/2)

(b::BasisFunction)(x) = b.shape(x, b.centre)
function (b::BasisFunction)(x, p::PeriodicBCHandler)
  in(x, b) && return b(x)
  in(x, translate(b, length(p))) && return translate(b, length(p))(x)
  in(x, translate(b,-length(p))) && return translate(b,-length(p))(x)
  return b(x)
end


function overlap(a::BasisFunction, b::BasisFunction)
  return lower(a) < upper(b) && upper(a) > lower(b)
end
Base.in(a::BasisFunction, b::BasisFunction) = overlap(a, b)

BasisFunction(centre::Number, width::Number) = BasisFunction(TopHat(width), centre)

function integral(b::BasisFunction, f, _::PeriodicBCHandler) where {F}
  return QuadGK.quadgk(x->b(x) * f(x), lower(b), upper(b))[1]
end

function integral(b::BasisFunction, limits::Union{Tuple, AbstractVector})
  lower, upper = limits
  @assert lower < upper
  return integral(b, lower, upper)[1]
end

function integral(a::BasisFunction{S1},
                  b::BasisFunction{S2},
                  p::PeriodicBCHandler) where {S1<:AbstractShape, S2<:AbstractShape}
  in(a, b) && return integral(a, b)
  in(translate(a, length(p)), b) && return integral(translate(a, length(p)), b)
  in(translate(a,-length(p)), b) && return integral(translate(a,-length(p)), b)
  in(a, translate(b, length(p))) && return integral(a, translate(b, length(p)))
  in(a, translate(b,-length(p))) && return integral(a, translate(b,-length(p)))
  return 0.0
end

function integral(a::BasisFunction{GaussianShape},
                  b::BasisFunction{TopHat})
  return (erf(-(a.centre - upper(b)) / a.shape.σ) -
          erf(-(a.centre - lower(b)) / a.shape.σ))/2
end
function integral(a::BasisFunction{TopHat},
                  b::BasisFunction{TopHat})
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

