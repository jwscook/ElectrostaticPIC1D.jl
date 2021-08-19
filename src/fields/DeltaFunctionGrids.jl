
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

cellcentre(g::DeltaFunctionGrid, i::Integer) = (i - 0.5) * cellwidth(g)
cellcentres(g::DeltaFunctionGrid) = (cellcentre(g, i) for i ∈ 1:g.N)
cellwidth(g::DeltaFunctionGrid) = g.L / g.N
cell(x, g::DeltaFunctionGrid) = Int(round(x / cellwidth(g)))
cellindices(ab, g::DeltaFunctionGrid) = cellindices(ab..., g)
function cellindices(a, b, g::DeltaFunctionGrid)
  return Int(floor(a / cellwidth(g))):Int(ceil(b / cellwidth(g)))
end
basis(g, i) = BasisFunction(TopHatShape(cellwidth(g)), cellcentre(g, i))
function Base.intersect(x::BasisFunction{S},
                        g::DeltaFunctionGrid{BC, T}) where {S, BC, T}
  indlo = Int(floor(lower(x) / cellwidth(g) + 0.5)) 
  indhi = Int(ceil(upper(x) / cellwidth(g) + 0.5))
  return (basis(g, mod1(i, g.N)) for i ∈ indlo:indhi)
end

# AbstractArray interface
Base.size(g::DeltaFunctionGrid) = (g.N,)
# DeltaFunctionGrid
Base.getindex(g::DeltaFunctionGrid, i) = g.values[i]
Base.setindex!(g::DeltaFunctionGrid, v, i) = (g.values[i] = v)
Base.iterate(g::DeltaFunctionGrid) = iterate(g.values)
Base.iterate(g::DeltaFunctionGrid, state) = iterate(g.values, state)

zero!(g::DeltaFunctionGrid) = (g.values .*= false)

