
struct EquispacedValueGrid{BC<:AbstractBC, T} <: AbstractGrid{BC, T}
  N::Int
  L::Float64
  values::Vector{T}
  bc::BC
end
function EquispacedValueGrid(N::Int, L::Real, ::Type{BC}=PeriodicGridBC,
   ::Type{T}=Float64) where {BC<:AbstractBC, T<:Number}
  return EquispacedValueGrid{BC,T}(N, L, zeros(T, N), BC(0.0, L))
end
Base.length(g::EquispacedValueGrid) = g.N
Base.ndims(::Type{EquispacedValueGrid{BC, T}}) where {BC<:AbstractBC, T} = 1

cellwidth(g::EquispacedValueGrid) = g.L / g.N
cell(x, g::EquispacedValueGrid) = Int(floor(x / cellwidth(g))) + 1
cellcentre(g::EquispacedValueGrid, i::Integer) = (i - 0.5) * cellwidth(g)
cellcentres(g::EquispacedValueGrid) = (cellcentre(g, i) for i ∈ 1:g.N)
cellindices(ab, g::EquispacedValueGrid) = cellindices(ab..., g)
numberofunknowns(g::EquispacedValueGrid) = g.N
function cellindices(a, b, g::EquispacedValueGrid)
  return Int(floor(a / cellwidth(g))):Int(ceil(b / cellwidth(g)))
end
basis(g, i) = BasisFunction(TopHatShape(cellwidth(g)), cellcentre(g, i))
cells(g) = [basis(g, i) for i ∈ 1:g.N]
function Base.intersect(x::BasisFunction{S, T1}, g::EquispacedValueGrid{BC, T2}
    ) where {S<:AbstractShape, BC, T1, T2}
  indlo = Int(floor(lower(x) / cellwidth(g) + 0.5)) 
  indhi = Int(ceil(upper(x) / cellwidth(g) + 0.5))
  return ((mod1(i, g.N), basis(g, mod1(i, g.N))) for i ∈ indlo:indhi)
end

function Base.isapprox(a::T, b::T, atol=0, rtol=sqrt(eps())) where {T<:EquispacedValueGrid}
  a.N == b.N && return false
  a.L == b.L && return false
  return isapprox(a.values, b.values, atol=atol, rtol=rtol)
end

# AbstractArray interface
Base.size(g::EquispacedValueGrid) = (g.N,)
# EquispacedValueGrid
Base.getindex(g::EquispacedValueGrid, i) = g.values[i]
Base.setindex!(g::EquispacedValueGrid, v, i) = (g.values[i] = v)
Base.iterate(g::EquispacedValueGrid) = iterate(g.values)
Base.iterate(g::EquispacedValueGrid, state) = iterate(g.values, state)

zero!(g::EquispacedValueGrid) = (g.values .*= false)

function deposit!(evg::EquispacedValueGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in evg that particle overlaps with
  bc = BC(0.0, evg.L)
  qw = charge(particle) * weight(particle)
  for (index, item) ∈ intersect(basis(particle), evg)
    # multiply by width because all basis functions are normalised
    amountincell = integral(item, basis(particle), bc) * width(item)
    evg[index] += amountincell * qw
  end
  return evg
end

function antideposit(evg::EquispacedValueGrid{BC}, particle) where {BC<:AbstractBC}
  # loop over all items in evg that particle overlaps with
  bc = BC(0.0, evg.L)
  amount = 0.0
  for (index, item) ∈ intersect(basis(particle), evg)
    # multiply by width because all basis functions are normalised
    amount += integral(item, basis(particle), bc) * width(item)
  end
  return evg
end

