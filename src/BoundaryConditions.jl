
abstract type AbstractBC end
abstract type AbstractPeriodicBC <: AbstractBC end
struct PeriodicParticleBC <: AbstractPeriodicBC end

abstract type AbstractBCHandler end

struct PeriodicBCHandler <: AbstractBCHandler
  lower::Float64
  upper::Float64
end
Base.length(p::PeriodicBCHandler) = p.upper - p.lower
Base.in(x::Number, p::PeriodicBCHandler) = p.lower <= x < p.upper
Base.:<(x::Number, p::PeriodicBCHandler) = !(p.lower <= x)
Base.:>(x::Number, p::PeriodicBCHandler) = !(x < p.upper)
Base.in(x, p::PeriodicBCHandler) = x[1] < p.upper && x[2] >= p.lower

function (p::PeriodicBCHandler)(x::Number)
  x ∈ p && return x
  x < p && return p(x + length(x))
  x > p && return p(x - length(x))
  throw(error("Should never reach here, $p, $x"))
end

