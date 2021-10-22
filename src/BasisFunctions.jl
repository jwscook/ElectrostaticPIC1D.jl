using FastGaussQuadrature, Memoization, MuladdMacro, SpecialFunctions

const RTOL = eps()

abstract type AbstractShape end
struct GaussianShape <: AbstractShape
  σ::Float64
  function GaussianShape(σ)
    σ > 0 || throw(error(ArgumentError("GuassianShape σ must be > 0 but is $σ")))
    return new(σ)
  end
end
width(s::GaussianShape) = 12.5 .* s.σ
sigma(s::GaussianShape) = s.σ
(s::GaussianShape)(x, centre) = exp(-(x-centre)^2 / s.σ^2) / √π / s.σ
knots(s::GaussianShape) = 0:1

struct BSpline{N} <: AbstractShape
  Δ::Float64
  function BSpline{N}(Δ) where {N}
    Δ > 0 || throw(error(ArgumentError("BSpline{$N} Δ must be > 0 but is $Δ")))
    return new{N}(Δ)
  end
end
width(s::BSpline{N}) where N = (N + 1) * s.Δ

const TopHatShape = BSpline{0}
const TentShape = BSpline{1}
const QuadraticBSplineShape = BSpline{2}
const CubicBSplineShape = BSpline{3}

(b::BSpline{N})(x, centre) where {N} = b((x - centre + width(b) / 2))
(b::BSpline{N})(x) where {N} = deboor(Unsigned(N), b.Δ, x) / b.Δ

@inline function deboor(degree::Unsigned, knots, x::Number; j::Integer=1)
  @assert length(knots) >= j + degree + 1
  return @inbounds if degree == 0
    knots[j] <= x < knots[j + 1] ? 1 : 0
  else
    @muladd (x - knots[j]) / (knots[j + degree] - knots[j]) * deboor(degree - 1, knots, x; j=j) +
    (knots[j + 1 + degree] - x) / (knots[j + 1 + degree] - knots[j + 1]) * deboor(degree - 1, knots, x; j=j + 1)
  end
end

@inline function deboor(d::Unsigned, t::Number, x::Number; j::Integer=1)
  ti = t * (j - 1)
  return if d == 0
    ti <= x < ti + t ? 1 : 0
  else
    td = t * d
    @muladd ((x - ti) * deboor(d - 1, t, x; j=j) + (ti + t + td - x) * deboor(d - 1, t, x; j=j + 1)) / td
  end
end


knots(b::BSpline{N}) where N = 0:1/(N + 1):1

struct DeltaFunctionShape <: AbstractShape end
DeltaFunctionShape(_) = DeltaFunctionShape() # so has same ctor as others
width(s::DeltaFunctionShape) = 0

mutable struct BasisFunction{S<:AbstractShape, T}
  shape::S
  centre::Float64
  weight::T
end
BasisFunction(s::AbstractShape, centre) = BasisFunction(s, centre, 0.0)

function Base.copy!(a::BasisFunction{S, T}, b::BasisFunction{S,T}) where {S,T}
  @assert a.shape == b.shape "copy! not intended for use between basisfunctions with non-like shapes"
  a.centre = b.centre
  a.weight = copy(b.weight)
  return a
end

lower(b::BasisFunction) = b.centre - width(b.shape) / 2
upper(b::BasisFunction) = b.centre + width(b.shape) / 2
lower(a::BasisFunction, b::BasisFunction) = max(lower(a), lower(b))
upper(a::BasisFunction, b::BasisFunction) = min(upper(a), upper(b))
width(b::BasisFunction) = width(b.shape)
weight(b::BasisFunction) = b.weight
centre(b::BasisFunction) = b.centre
shape(b::BasisFunction) = b.shape
knots(b::BasisFunction) = knots(shape(b)) .* width(b) .+ lower(b)
function getkey(u::BasisFunction, v::BasisFunction)::UInt64
  hashtuple = (width(u), width(v), trunc((centre(u) - centre(v)), digits=14))
  return foldr(hash, hashtuple; init=hash(typeof(u), hash(typeof(v))))
end
function translate(b::BasisFunction, x::Number)
  @assert isfinite(x)
  translated = deepcopy(b)
  translated.centre += x
  return translated
end
function translate!(b::BasisFunction, x::Number, bc::AbstractBC)
  b.centre = bc(b.centre + x)
  return b
end

function Base.:+(b::BasisFunction, x)
  @assert isfinite(x) "x is $x"
  b.weight += x
  return b
end
for op in (:-, :+)
  @eval function Base.$op(a::BasisFunction, b::BasisFunction)
    @assert centre(a) == centre(b)
    a.weight = $op(a.weight, b.weight)
    return a
  end
end
function Base.isapprox(a::BasisFunction{S}, b::BasisFunction{S};
    atol=0.0, rtol=sqrt(eps())) where {S}
  @assert centre(a) == centre(b)
  return isapprox(a.weight, b.weight, atol=atol, rtol=rtol)
end

Base.isfinite(b::BasisFunction) = isfinite(b.centre) && isfinite(b.weight)


function blend(basisfunctions::NTuple{N, BasisFunction{S, T}}, factors) where {N, S, T}
  @assert N == length(factors)
  c, w = 0.0, zero(T)
  for (b, f) in zip(basisfunctions, factors)
    @muladd c = c + b.centre * f
    @muladd w = w + b.weight * f
  end
  return BasisFunction(basisfunctions[1].shape, c, w)
end


zero!(b::BasisFunction) = (b.weight *= false)
function Base.in(x::Number, b::BasisFunction)
  return (b.centre - width(b)/2 <= x < b.centre + width(b)/2)
end

function knots(a::BasisFunction, b::BasisFunction)
  @assert overlap(a, b)
  ka = knots(a)
  kb = knots(b)
  lo = max(ka[1], kb[1])
  hi = min(ka[end], kb[end])
  output = filter(x->lo <= x <= hi, unique(vcat(ka, kb)))
  sort!(output)
  return output
end

function translate(a::BasisFunction{S1}, b::BasisFunction{S2},
    p::PeriodicGridBC) where {S1<:AbstractShape, S2<:AbstractShape}
  lower(p) <= lower(a) && upper(a) < upper(p) && lower(p) <= lower(b) && upper(b) < upper(p) && return (a, b) 
  in(a, b) && return (a, b)
  t = translate(a, length(p)); in(t, b) && return (t, b)
  t = translate(a,-length(p)); in(t, b) && return (t, b)
  return (a, b)
end

(b::BasisFunction)(x::Number) = b.shape(x, b.centre)
function (b::BasisFunction)(x::Number, p::PeriodicGridBC)
  in(x, b) && return b(x)
  t = translate(b, length(p)); in(x, t) && return t(x)
  t = translate(b,-length(p)); in(x, t) && return t(x)
  return b(x)
end


function overlap(a::BasisFunction, b::BasisFunction)
  return lower(a) == lower(b) || (lower(a) < upper(b) && upper(a) > lower(b))
end
Base.in(a::BasisFunction, b::BasisFunction) = overlap(a, b)
function overlap(u::BasisFunction, v::BasisFunction, p::PeriodicGridBC)
  return in(translate(u, v, p)...)
end


function integral(b::BasisFunction{DeltaFunctionShape}, f::F, _::AbstractBC) where {F<:Function}
  return f(centre(b))
end

function integral(b::BasisFunction, f::F, _::AbstractBC) where {F}
  ks = knots(b)
  return mapreduce(i->
    QuadGK.quadgk(x->b(x) * f(x), ks[i], ks[i+1]; order=11, atol=eps(), rtol=RTOL)[1],
    +, 1:length(ks)-1)
end

function integral(u::BasisFunction{S1},
                  v::BasisFunction{S2},
                  p::PeriodicGridBC) where {S1<:AbstractShape, S2<:AbstractShape}
  u, v = translate(u, v, p)
  return in(u, v) ? integral(u, v) : 0.0
end

function integral(u::BasisFunction{BSpline{N1}}, v::BasisFunction{BSpline{N2}}
    ) where {N1, N2}
  output = 0.0
  kn = knots(u, v)
  x, w = FastGaussQuadrature.gausslegendre((N1 + 1) * (N2 + 1))
  for i ∈ 1:length(kn)-1
    a, b = kn[i], kn[i+1]
    for j ∈ eachindex(x)
      xj = (x[j] + 1) / 2 * (b - a) + a
      wj = w[j] / 2 * (b - a)
      output += wj * u(xj) * v(xj)
    end
  end
  return output
end
function integral(u::BasisFunction{<:AbstractShape, T1},
                  v::BasisFunction{<:AbstractShape, T2}) where {T1, T2}
  ks = knots(u, v)
  return mapreduce(i->
    QuadGK.quadgk(x->u(x) * v(x), ks[i], ks[i+1]; order=11, atol=eps(), rtol=RTOL)[1],
    +, 1:length(ks)-1)
end

function integral(u::BasisFunction{GaussianShape}, v::BasisFunction{TopHatShape})
  return erf(-(u.centre - lower(v)) / u.shape.σ,
             -(u.centre - upper(v)) / u.shape.σ) / 2 / width(v)
end
function integral(u::BasisFunction{TopHatShape},
                  v::BasisFunction{TopHatShape})
  w = upper(u, v) - lower(u, v)
  h = 1 / (u.shape.Δ * v.shape.Δ)
  return w * h * overlap(u, v)
end
integral(a::BasisFunction{DeltaFunctionShape}, b::BasisFunction) = b(a.centre)
integral(a::BasisFunction, b::BasisFunction{DeltaFunctionShape}) = a(b.centre)

function integral(a::BasisFunction{GaussianShape},
                  b::BasisFunction{GaussianShape})
  σ₁, σ₂ = a.shape.σ, b.shape.σ
  μ₁, μ₂ = a.centre, b.centre
  σ² = σ₁^2 + σ₂^2
  return exp(-(μ₁ - μ₂)^2 / σ²) / √(π * σ²)
end


function integral(a::BasisFunction{GaussianShape}, b::BasisFunction{TentShape})
  σ = a.shape.σ
  μ = centre(a)
  Δ = b.shape.Δ
  c = centre(b)
  foo(l, m) = @muladd (σ * exp(-(m - l)^2/σ^2)/√π + m * erf((m - l)/σ)) / 2Δ^2
  return foo(0, μ - c + Δ) - foo(Δ, μ - c + Δ) +
    foo(0, μ - c - Δ) - foo(-Δ, μ - c - Δ)
end


