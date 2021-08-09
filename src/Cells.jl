mutable struct Cell{T}
  lower::Float64
  upper::Float64
  value::T
end
Cell(lower::Float64, upper::Float64) = Cell{Float64}(lower, upper, 0.0)

centre(c::Cell) = (c.lower .+ c.upper) ./ 2
support(c::Cell) = (c.lower, c.upper)
Base.in(x, c::Cell) = all((x .>= c.lower) && (x .<= c.upper))
Base.zero!(c::Cell) = (c.value *= 0);
Base.+=(c::Cell, x) = (c.value += x);
Base.*=(c::Cell, x) = (c.value *= x);


