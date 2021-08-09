@concrete struct PeriodicFourierWorkspace
  workvector::Vector{ComplexF64}
  ik::Vector{Float64}
  fft_plan
end
function PeriodicFourierWorkspace(N::Int, L::Float64)
  @assert iseven(N)
  ik = 2π * im * vcat(0:N÷2,-N÷2+1:-1) ./ L; 
  return PeriodicFourierWorkspace(zeros(ComplexF64, N), ik, nothing)
end

struct FourierField{BC<:PeriodicGridBC, T} <: AbstractField{BC}
  charge::DeltaFunctionGrid{BC,T}
  electricfield::DeltaFunctionGrid{BC,T}
  helper::PeriodicFourierWorkspace
end
function FourierField(charge::DeltaFunctionGrid{PeriodicGridBC})
  electricfield = deepcopy(charge)
  zero!(electricfield)
  return FourierField(charge, electricfield,
    PeriodicFourierWorkspace(size(charge), charge.L))
end

function solve!(f::FourierField)
  f.helper.workvector .= f.charge ./ f.helper
  fft!(f.helper.workvector)
  f.helper.workvector[1] *= false
  f.electricfield .= real.(ifft!(f.helper.workvector))
  return nothing
end

