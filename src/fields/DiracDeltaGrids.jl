
struct DeltaFunctionGrid{BC<:AbstractBC, T} <: AbstractGrid{BC}
  N::Int
  L::Float64
  bc::BC
  cells::Vector{Cell{T}}
end
function DeltaFunctionGrid(N::Int, L::Int, ::Type{BC}=PeriodicGridBC,
   ::Type{T}=Float64) where {BC<:AbstractBC, T<:Number}
  Δ = L / N
  cells = collect(Cell((i-1) * Δ, i * Δ, 0.0) for i ∈ 1:N)
  return DeltaFunctionGrid{BC,T}(N, L, BC(N, L), cells)
end
cellcentres(g::DeltaFunctionGrid) = ((0:g.N) .+ 0.5) * g.L
cellwidth(g::DeltaFunctionGrid) = L / g.N
cell(x, g::DeltaFunctionGrid) = Int(round(x / g.L * grid.N))
cell(p::Particle, g::DeltaFunctionGrid) = cell(p.x, g)
#celledges(x, g::DeltaFunctionGrid) = cell(x, g) .+ (-1/2, 1/2) .* cellwidth(g)
cells(ab, g::DeltaFunctionGrid) = cells(ab..., g)
function cells(a, b, g::DeltaFunctionGrid)
  indices = Int(floor(a / cellwidth(g))):Int(ceil(b / cellwidth(g)))
  return @view g.cells[indices]
end
Base.in(x::Number, g::DeltaFunctionGrid) = cells(x[1], x[2], g)
Base.in(x, g::DeltaFunctionGrid) = cells(x[1], x[2], g)

# AbstractArray interface
Base.size(g::AbstractGrid) = g.N
# DeltaFunctionGrid
Base.getindex(g::DeltaFunctionGrid, i) = g.cells[i]
Base.setindex!(g::DeltaFunctionGrid, v, i) = (g.cells[i] = v)
Base.iterate(g::DeltaFunctionGrid) = iterate(g.cells)
Base.iterate(g::DeltaFunctionGrid, state) = iterate(g.cells, state)

function deposit!(g::DeltaFunctionGrid, particle) where {F}
  for cell ∈ cells(support(particle), g)
    cell += weight(particle) * charge(particle, cell) * particle.weight
  end
  return nothing
end

#(g::DeltaFunctionGrid)(i::Int) = i * g.L / g.N;
#(g::DeltaFunctionGrid)(c::AbstractFloat) = round(c / g.L * g.N);

