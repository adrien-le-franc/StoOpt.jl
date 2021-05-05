# developed with Julia 1.4.2
#
# generic struct for Stochastic Optimization problems 


# Types for value functions

abstract type ValueFunctions end

mutable struct ArrayValueFunctions <: ValueFunctions
	functions::Array{Float64}
end

ArrayValueFunctions(sz::Vararg{Int64}) = ArrayValueFunctions(zeros(sz))

Base.:(==)(vf1::ArrayValueFunctions, vf2::ArrayValueFunctions) = (vf1.functions == vf2.functions)
Base.size(vf::ArrayValueFunctions) = size(vf.functions)
Base.getindex(vf::ArrayValueFunctions, t::Int64) = vf.functions[t, ..]
Base.setindex!(vf::ArrayValueFunctions, x::Array{Float64}, t::Int64) = (vf.functions[t, ..] = x)
Base.getindex(vf::ArrayValueFunctions, index::Vararg{Int64}) = vf.functions[index...]
function Base.setindex!(vf::ArrayValueFunctions, x::Float64, 
	index::Vararg{Int64}) vf.functions[index...] = x end


# Types for interpolating ValueFunctions on States

mutable struct Interpolator{N, Tv <: AbstractArray{Float64,N}}
	value::Tv
end

function Interpolator(t::Int64, states::States, value_functions::ValueFunctions)
	return Interpolator(LinearInterpolation(tuple(states.axis...), value_functions[t]))
end

function update_interpolator!(interpolator::Interpolator, t::Int64, 
	states::States, value_functions::ValueFunctions)
	interpolator.value = LinearInterpolation(tuple(states.axis...), value_functions[t])
end