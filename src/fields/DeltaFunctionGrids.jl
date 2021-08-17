
struct DeltaFunctionGrid{BC<:AbstractBC, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  values::Vector{T}
  bc::BC
end
function DeltaFunctionGrid(N::Int, L::Real, ::Type{BC}=PeriodicGridBC,
   ::Type{T}=Float64) where {BC<:AbstractBC, T<:Number}
  return DeltaFunctionGrid{BC,T}(N, L, zeros(T, N), BC(0.0, L))
end
Base.length(g::DeltaFunctionGrid) = g.N
Base.ndims(::Type{DeltaFunctionGrid{BC, T}}) where {BC<:AbstractBC, T} = 1

cellcentres(g::DeltaFunctionGrid) = ((1:g.N) .- 0.5) * cellwidth(g)
cellwidth(g::DeltaFunctionGrid) = g.L / g.N
cell(x, g::DeltaFunctionGrid) = Int(round(x / cellwidth(g)))
cell(p::Particle, g::DeltaFunctionGrid) = cell(p.x, g)
#celledges(x, g::DeltaFunctionGrid) = cell(x, g) .+ (-1/2, 1/2) .* cellwidth(g)
cells(ab, g::DeltaFunctionGrid) = cells(ab..., g)
function cells(a, b, g::DeltaFunctionGrid)
  return Int(floor(a / cellwidth(g))):Int(ceil(b / cellwidth(g)))
end
Base.in(x::Number, g::DeltaFunctionGrid) = cells(x, g)
Base.in(x, g::DeltaFunctionGrid) = cells(x[1], x[2], g)

# AbstractArray interface
Base.size(g::DeltaFunctionGrid) = (g.N,)
# DeltaFunctionGrid
Base.getindex(g::DeltaFunctionGrid, i) = g.values[i]
Base.setindex!(g::DeltaFunctionGrid, v, i) = (g.values[i] = v)
Base.iterate(g::DeltaFunctionGrid) = iterate(g.values)
Base.iterate(g::DeltaFunctionGrid, state) = iterate(g.values, state)

zero!(g::DeltaFunctionGrid) = (g.values .*= false)
function deposit!(g::DeltaFunctionGrid, particle) where {F}
  for cell ∈ cells(support(particle), g)
    cell += weight(particle) * charge(particle, cell) * particle.weight
  end
  return nothing
end

