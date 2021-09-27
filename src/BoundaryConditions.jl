
abstract type AbstractBC end

# Periodic BC
struct PeriodicGridBC <: AbstractBC
  lower::Float64
  upper::Float64
end
PeriodicGridBC(x) = PeriodicGridBC(0.0, x)
lower(p::PeriodicGridBC) = p.lower
upper(p::PeriodicGridBC) = p.upper

Base.length(p::PeriodicGridBC) = p.upper - p.lower
Base.in(x::Number, p::PeriodicGridBC) = p.lower <= x < p.upper
Base.:<(x::Number, p::PeriodicGridBC) = x < p.lower
Base.:>(x::Number, p::PeriodicGridBC) = x > p.upper

function (p::PeriodicGridBC)(x::Number)
  @assert isfinite(x)
  x ∈ p && return x
  x < p && return p(x + length(p))
  x > p && return p(x - length(p))
  @show x, p
  throw(error("Should never reach here, $p, $x"))
end

