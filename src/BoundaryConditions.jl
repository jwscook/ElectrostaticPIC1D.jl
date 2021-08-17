
abstract type AbstractBC end

# Periodic BC
struct PeriodicGridBC <: AbstractBC
  lower::Float64
  upper::Float64
end

Base.length(p::PeriodicGridBC) = p.upper - p.lower
Base.in(x::Number, p::PeriodicGridBC) = p.lower <= x < p.upper
Base.:<(x::Number, p::PeriodicGridBC) = !(p.lower <= x)
Base.:>(x::Number, p::PeriodicGridBC) = !(x < p.upper)
Base.in(x, p::PeriodicGridBC) = x[1] < p.upper && x[2] >= p.lower

function (p::PeriodicGridBC)(x::Number)
  x ∈ p && return x
  x < p && return p(x + length(x))
  x > p && return p(x - length(x))
  throw(error("Should never reach here, $p, $x"))
end

