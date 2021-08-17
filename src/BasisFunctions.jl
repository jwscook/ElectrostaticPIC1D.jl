using Memoization, SpecialFunctions

abstract type AbstractShape end
struct GaussianShape <: AbstractShape
  σ::Float64
  function GaussianShape(σ)
    σ > 0 || throw(error(ArgumentError("GuassianShape σ must be > 0 but is $σ")))
    return new(σ)
  end
end
width(s::GaussianShape) = 13 .* s.σ
(s::GaussianShape)(x, centre) = exp(-(x-centre)^2 / s.σ^2) / √π / s.σ

struct BSpline{N} <: AbstractShape
  Δ::Float64
  function BSpline{N}(Δ) where {N}
    Δ > 0 || throw(error(ArgumentError("BSpline{$N} Δ must be > 0 but is $Δ")))
    return new{N}(Δ)
  end
end
width(s::BSpline{N}) where N = (N + 1) * s.Δ

function (s::BSpline{0})(x, centre)
  return (-s.Δ/2 <= x - centre < s.Δ/2) / s.Δ
end
const TopHatShape = BSpline{0}

function (s::BSpline{1})(x, centre)
  z = 2 * (x - centre + 1 * s.Δ) / width(s) # z is between 0 and 2
  value = if 0 <= z < 1
    z
  elseif 1 <= z < 2
    (2 - z)
  else
    zero(z)
  end
  return value / s.Δ
end
const TentShape = BSpline{1}

function (s::BSpline{2})(x, centre)
  z = 3 * (x - centre + 1.5 * s.Δ) / width(s) # z is between 0 and 3
  value = if 0 <= z < 1
    z^2 / 2
  elseif 1 <= z < 2
    3/4 - (1.5 - z)^2
  elseif 2 <= z < 3
    (3 - z)^2 / 2
  else
    zero(z)
  end
  return value / s.Δ
end
const QuadraticBSplineShape = BSpline{2}


struct DeltaFunctionShape <: AbstractShape end
DeltaFunctionShape(_) = DeltaFunctionShape() # so has same ctor as others
width(s::DeltaFunctionShape) = 0
(s::DeltaFunctionShape)(x, centre) = x == centre

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
function getkey(u::BasisFunction, v::BasisFunction)::UInt64
  hashtuple = (width(u), width(v), trunc((centre(u) - centre(v)), digits=14))
  return foldr(hash, hashtuple; init=hash(typeof(u), hash(typeof(v))))
end
function translate(b::BasisFunction, x::Number)
  translated = deepcopy(b)
  translated.centre += x
  return translated
end
Base.:+(b::BasisFunction, x) = (b.weight += x; b)
Base.:*(x, b::BasisFunction) = x * b.weight
Base.:*(b::BasisFunction, x) = x * b.weight
zero!(b::BasisFunction) = (b.weight *= false)
Base.in(x, b::BasisFunction) = (b.centre - width(b)/2 <= x < b.centre + width(b)/2)

function translate(a::BasisFunction{S1}, b::BasisFunction{S2},
    p::PeriodicBCHandler) where {S1<:AbstractShape, S2<:AbstractShape}
  in(a, b) && return (a, b)
  in(translate(a, length(p)), b) && return (translate(a, length(p)), b)
  in(translate(a,-length(p)), b) && return (translate(a,-length(p)), b)
  return (a, b)
end

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
#  in(a, translate(b, length(p))) && return integral(a, translate(b, length(p)))
#  in(a, translate(b,-length(p))) && return integral(a, translate(b,-length(p)))
  return 0.0
end

function integral(a::BasisFunction{GaussianShape},
                  b::BasisFunction{TopHatShape})
  return (erf(-(a.centre - upper(b)) / a.shape.σ) -
          erf(-(a.centre - lower(b)) / a.shape.σ))/2
end
function integral(a::BasisFunction{TopHatShape},
                  b::BasisFunction{TopHatShape})
  return (min(upper(a), upper(b)) - max(lower(a), lower(b))) * overlap(a, b)
end
integral(a::BasisFunction{DeltaFunctionShape}, b::BasisFunction) = b(a.centre)
integral(a::BasisFunction, b::BasisFunction{DeltaFunctionShape}) = a(b.centre)

function integral(a::BasisFunction{GaussianShape},
                  b::BasisFunction{GaussianShape})
  σ₁, σ₂ = a.shape.σ, b.shape.σ
  μ₁, μ₂ = a.centre, b.centre
  σ² = σ₁^2 + σ₂^2
  return √π * σ₁ * σ₂ / σ² * exp(-(μ₁ - μ₂)^2 / σ²)
end

