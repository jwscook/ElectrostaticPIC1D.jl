@concrete struct PeriodicFourierWorkspace
  workvector::Vector{ComplexF64}
  ik::Vector{ComplexF64}
  fft_plan
end

function PeriodicFourierWorkspace(N::Int, L::Float64)
  @assert iseven(N)
  ik = 2π * im * vcat(0:N÷2,-N÷2+1:-1) ./ L; 
  return PeriodicFourierWorkspace(zeros(ComplexF64, N), ik, nothing)
end
Base.length(p::PeriodicFourierWorkspace) = length(p.ik)

struct FourierField{BC<:PeriodicGridBC, T} <: AbstractField{BC}
  charge::EquispacedValueGrid{BC,T}
  electricfield::EquispacedValueGrid{BC,T}
  helper::PeriodicFourierWorkspace
end
function FourierField(N::Int, L::Real)
  return FourierField(EquispacedValueGrid(N, L, PeriodicGridBC))
end
function FourierField(charge::EquispacedValueGrid{PeriodicGridBC})
  electricfield = deepcopy(charge)
  zero!(electricfield)
  return FourierField(charge, electricfield,
    PeriodicFourierWorkspace(first(size(charge)), charge.L))
end

zero!(f::FourierField) = (zero!(f.charge), zero!(f.electricfield))

function solve!(f::FourierField)
  f.helper.workvector .= f.charge
  fft!(f.helper.workvector)
  f.helper.workvector ./= f.helper.ik
  f.helper.workvector[1] *= false
  f.electricfield .= real.(ifft!(f.helper.workvector))
  return nothing
end

Base.getindex(g::FourierField, i) = g.values[i]
Base.setindex!(g::FourierField, v, i) = (g.values[i] = v)
Base.iterate(g::FourierField) = iterate(g.values)
Base.iterate(g::FourierField, state) = iterate(g.values, state)

