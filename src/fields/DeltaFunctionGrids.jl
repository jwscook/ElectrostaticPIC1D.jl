
struct DeltaFunctionGrid{BC<:AbstractBC, T} <: AbstractGrid{BC}
  N::Int
  L::Float64
  cells::Vector{Cell{T}}
  bc::BC
end
function DeltaFunctionGrid(N::Int, L::Real, ::Type{BC}=PeriodicGridBC,
   ::Type{T}=Float64) where {BC<:AbstractBC, T<:Number}
  Δ = L / N
  cells = collect(Cell((i-1) * Δ, i * Δ, 0.0) for i ∈ 1:N)
  return DeltaFunctionGrid{BC,T}(N, L, cells, BC())
end
Base.length(g::DeltaFunctionGrid) = g.N
Base.ndims(::Type{DeltaFunctionGrid{BC, T}}) where {BC<:AbstractBC, T} = 1
function Base.isapprox(g::DeltaFunctionGrid, x::Vector{Float64}; kwargs...)
  return all(.≈(g.cells, x, kwargs...))
end


cellcentres(g::DeltaFunctionGrid) = ((1:g.N) .- 0.5) * cellwidth(g)
cellwidth(g::DeltaFunctionGrid) = g.L / g.N
cell(x, g::DeltaFunctionGrid) = Int(round(x / cellwidth(g)))
cell(p::Particle, g::DeltaFunctionGrid) = cell(p.x, g)
#celledges(x, g::DeltaFunctionGrid) = cell(x, g) .+ (-1/2, 1/2) .* cellwidth(g)
cells(ab, g::DeltaFunctionGrid) = cells(ab..., g)
function cells(a, b, g::DeltaFunctionGrid)
  indices = Int(floor(a / cellwidth(g))):Int(ceil(b / cellwidth(g)))
  return @view g.cells[indices]
end
Base.in(x::Number, g::DeltaFunctionGrid) = cells(x[1], x[2], g)
Base.in(x, g::DeltaFunctionGrid) = cells(x[1], x[2], g)

function Base.copyto!(g::DeltaFunctionGrid, v)
  for (i, vi) ∈ enumerate(v)
    zero!(g[i])
    g[i] += vi
  end
  return g
end

# AbstractArray interface
Base.size(g::DeltaFunctionGrid) = (g.N,)
# DeltaFunctionGrid
Base.getindex(g::DeltaFunctionGrid, i) = g.cells[i]
Base.setindex!(g::DeltaFunctionGrid, v::Real, i) = (g.cells[i].value = v)
Base.setindex!(g::DeltaFunctionGrid, v, i) = (g.cells[i] = v)
Base.iterate(g::DeltaFunctionGrid) = iterate(g.cells)
Base.iterate(g::DeltaFunctionGrid, state) = iterate(g.cells, state)

zero!(g::DeltaFunctionGrid) = map(zero!, g)
function deposit!(g::DeltaFunctionGrid, particle) where {F}
  for cell ∈ cells(support(particle), g)
    cell += weight(particle) * charge(particle, cell) * particle.weight
  end
  return nothing
end

