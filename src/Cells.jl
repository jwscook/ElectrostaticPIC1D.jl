mutable struct Cell{T}
  lower::Float64
  upper::Float64
  value::T
end
Cell(lower::Float64, upper::Float64) = Cell{Float64}(lower, upper, 0.0)

centre(c::Cell) = (c.lower .+ c.upper) ./ 2
support(c::Cell) = (c.lower, c.upper)
value(c::Cell) = c.value
Base.in(x, c::Cell) = all((x .>= c.lower) && (x .<= c.upper))
zero!(c::Cell) = (c.value *= 0)
Base.:+(c::Cell, x) = (c.value += x)
Base.:*(c::Cell, x) = (c.value *= x)
Base.:/(c::Cell, x) = c.value / x
Base.isapprox(c::Cell, x::Float64; kwargs...) = isapprox(c.value, x, kwargs...)

#Base.convert(::T, c::Cell) where {T} = T(c.value)
Base.convert(::Cell{T}, c::Cell{T}) where {T} = c
Base.convert(::Cell{T}, c::Cell{U}) where {T,U} = Cell{T}(c.lower, c.upper, c.T(c.value))



