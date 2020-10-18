# developed with Julia 1.4.2
#
# generic struct for Stochastic Optimization problems 


# Type for bounds

struct Bounds
	upper::Array{Array{Float64,1},1}
	lower::Array{Array{Float64,1},1}

	function Bounds(upper::Array{Array{Float64,1},1}, lower::Array{Array{Float64,1},1})
		if length(upper) != length(lower)
			error("Bounds: length of upper $(length(upper)) != lower $(length(lower))")
		end
		for t in 1:length(upper)
			if !all(lower[t] .<= upper[t])
				error("Bounds: empty domain at step $(t), 
					where lower $(lower[t]) and upper $(upper[t])")
			end
		end
		new(upper, lower)
	end
end


Base.size(bounds::Bounds) = length(bounds.upper)
Base.getindex(bounds::Bounds, index::Int64) = bounds.upper[index], bounds.lower[index]

function Bounds(horizon::Int64, upper::Array{Float64,1}, lower::Array{Float64,1})
	return Bounds(fill(upper, horizon), fill(lower, horizon))
end

function Bounds(horizon::Int64, upper::Float64, lower::Float64)
	return Bounds(fill([upper], horizon), fill([lower], horizon))
end

# Type for states

struct States{Ta <: StepRangeLen{Float64}, Ti <: Iterators.Zip}
	axis::Array{Ta}
	iterator::Ti
	bounds::Bounds
end


function States(horizon::Int64, axis::Vararg{T}) where T <: StepRangeLen{Float64}

	axis = collect(axis)
	upper = Float64[]
	lower = Float64[]
	indices = []

	for i in 1:length(axis)
		x = axis[i]
		x_min, x_max = extrema(x)
		push!(upper, x_max)
		push!(lower, x_min)
		push!(indices, 1:length(x))
	end

	indices = Iterators.product(indices...)
	iterator = zip(Iterators.product(axis...), indices)
	bounds = Bounds(horizon+1, upper, lower)

	return States(axis, iterator, bounds)

end

Base.size(states::States) = size(states.iterator)

# Type for controls

struct Controls{T <: Iterators.ProductIterator}
	iterators::Array{T,1}
end


Base.size(controls::Controls) = length(controls.iterators)
Base.getindex(controls::Controls, index::Int64) = controls.iterators[index]

function Controls(horizon::Int64, axis::Vararg{T}) where T <: StepRangeLen{Float64}
	return Controls(fill(Iterators.product(axis...), horizon))
end

function Controls(bounds::Bounds, axis::Vararg{T}) where T <: StepRangeLen{Float64} 

	dimension = length(axis)
	iterators = Iterators.ProductIterator[]

	for t in 1:length(bounds.upper)

		upper, lower = bounds[t]
		axis_iterators = []

		for i in 1:dimension
			in_bounds(control::Float64) = lower[i] <= control <= upper[i]
			push!(axis_iterators, Iterators.filter(in_bounds, axis[i]))
		end

		push!(iterators, Iterators.product(axis_iterators...))

	end

	return Controls(iterators)

end

# Types for handling noise processes 

struct TimeProcess
	data::Array{Float64,3}
	horizon::Int64
	cardinal::Int64
	dimension::Int64

	function TimeProcess(x::Array{Float64,3})
		horizon, cardinal, dimension = size(x)
		new(x, horizon, cardinal, dimension)
	end
end

Base.size(tp::TimeProcess) = (tp.horizon, tp.cardinal, tp.dimension)
Base.getindex(tp::TimeProcess, t::Int64) = tp.data[t, :, :]
iterator(tp::TimeProcess, t::Int64) = eachrow(tp[t])

function TimeProcess(x::Array{Float64,2})
	horizon, cardinal = size(x)
	x = reshape(x, horizon, cardinal, 1)
	TimeProcess(x)
end


struct Probability
	data::Array{Float64,2}
	horizon::Int64
	cardinal::Int64

	function Probability(x::Array{Float64,2})
		horizon, cardinal = size(x)
		new(x, horizon, cardinal)
	end
end

Base.size(p::Probability) = (p.horizon, p.cardinal)
Base.getindex(p::Probability, t::Int64) = p.data[t, :]
iterator(p::Probability, t::Int64) = Iterators.Stateful(p[t])


struct Noises
	w::TimeProcess
	pw::Probability

	function Noises(w::TimeProcess, pw::Probability)
		h_w, c_w, _ = size(w)
		h_pw, c_pw = size(pw)
		if (h_w, c_w) != (h_pw, c_pw)
			error("Noises: noise size $(size(w)) not compatible with probabilities size $(size(pw))")
		end
		new(w, pw)
	end
end

Noises(w::Array{Float64}, pw::Array{Float64}) = Noises(TimeProcess(w), Probability(pw))

function Noises(data::Array{Float64,2}, k::Int64) 

	"""dicretize noise space to k values using Kmeans: return type Noises
	data > time series data of dimension (horizon, n_data)
	k > Kmeans parameter

	"""

	horizon, n_data = size(data)
	w = zeros(horizon, k)
	pw = zeros(horizon, k)

	for t in 1:horizon
		w_t = reshape(data[t, :], (1, :))
		kmeans_w = kmeans(w_t, k)
		w[t, :] = kmeans_w.centers
		pw[t, :] = kmeans_w.counts / n_data
	end

	return Noises(w, pw)

end


struct RandomVariable
	value::Array{Float64,2}
	probability::Array{Float64,1}
end

RandomVariable(n::Noises, t::Int64) = RandomVariable(n.w[t], n.pw[t])
iterator(rv::RandomVariable) = zip(eachrow(rv.value), Iterators.Stateful(rv.probability))

# Type containing all variables for Stochastic Optimal Control

mutable struct Variables
	t::Int64
	state::Union{Array{Float64,1}, Nothing}
	control::Union{Array{Float64,1}, Nothing}
	noise::Union{RandomVariable, Nothing}
end

Variables(t::Int64, rv::RandomVariable) = Variables(t, nothing, nothing, rv)

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


mutable struct SharedArrayValueFunctions <: ValueFunctions
	functions::SharedArray{Float64}
end

function SharedArrayValueFunctions(sz::Vararg{Int64})
	return SharedArrayValueFunctions(SharedArray{Float64, length(sz)}(sz)) end

function Base.:(==)(svf1::SharedArrayValueFunctions, svf2::SharedArrayValueFunctions) 
	return svf1.functions == svf2.functions end
Base.size(svf::SharedArrayValueFunctions) = size(svf.functions)
Base.getindex(svf::SharedArrayValueFunctions, t::Int64) = svf.functions[t, ..]
function Base.setindex!(svf::SharedArrayValueFunctions, x::Array{Float64}, t::Int64) 
	(svf.functions[t, ..] = x) end
Base.getindex(svf::SharedArrayValueFunctions, index::Vararg{Int64}) = svf.functions[index...]
function Base.setindex!(svf::SharedArrayValueFunctions, x::Float64, index::Vararg{Int64}) 
	(svf.functions[index...] = x) end

function ArrayValueFunctions(svf::SharedArrayValueFunctions)
	return ArrayValueFunctions(sdata(svf.functions)) end

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