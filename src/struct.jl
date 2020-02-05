# developed with Julia 1.1.1
#
# generic struct for Stochastic Optimization problems 


# Types for value functions


abstract type ValueFunctions end


mutable struct ArrayValueFunctions <: ValueFunctions
	functions::Array{Float64}
end

Base.getindex(vf::ArrayValueFunctions, t::Int64) = vf.functions[t, ..]
Base.setindex!(vf::ArrayValueFunctions, x::Array{Float64}, t::Int64) = (vf.functions[t, ..] = x)
Base.:(==)(vf1::ArrayValueFunctions, vf2::ArrayValueFunctions) = (vf1.functions == vf2.functions)
Base.size(vf::ArrayValueFunctions) = size(vf.functions)
ArrayValueFunctions(t::Tuple{Vararg{Int64}}) = ArrayValueFunctions(zeros(t))


mutable struct Cuts
	n_cuts::Int64
	alpha::Array{Array{Float64,1}}
	beta::Array{Float64,1}
end

Cuts() = Cuts(0, Array{Float64,1}[], Float64[])


mutable struct CutsValueFunctions <: ValueFunctions
	functions::Array{Cuts}
end

CutsValueFunctions(n::Int64) = CutsValueFunctions([Cuts() for t in 1:n])




# Type for discretized spaces 


struct Grid{T <: StepRangeLen{Float64}}
	axis::Array{T}
	iterator::Union{Iterators.ProductIterator, Iterators.Zip}
	steps::Array{Float64}
end

Base.getindex(g::Grid, i::Int) = g.axis[i]
Base.size(g::Grid) = Tuple([length(g[i]) for i in 1:length(g.axis)])

function Grid(axis::Vararg{T}; enumerate=false) where T <: StepRangeLen{Float64}
	axis = collect(axis)
	dimension = length(axis)
	grid_steps = [axis[i][2] - axis[i][1] for i in 1:dimension]
	if enumerate == true
		grid_size = [length(axis[i]) for i in 1:length(axis)]
		indices = Iterators.product([1:i for i in grid_size]...)
		return Grid(axis, zip(Iterators.product(axis...), indices), grid_steps)
	else
		return Grid(axis, Iterators.product(axis...), grid_steps)
	end
end


# Types for continuous spaces


struct Bounds
	lower_bounds::Array{Float64,1}
	upper_bounds::Array{Float64,1}
	n_variables::Int64

	function Bounds(lower_bounds::Array{Float64,1}, upper_bounds::Array{Float64,1})
		
		if length(upper_bounds) != length(lower_bounds)
			error("Bounds: lengths of upper and lower bounds do not match")
		end
		if !all(lower_bounds .<= upper_bounds)
			error("Bounds: lower_bounds .<= upper_bounds is not true")
		end
		new(lower_bounds, upper_bounds, length(upper_bounds))
	end
end


struct PolyhedralCost
	n_cuts::Int64
	update_cost!::Function
end


struct LinearDynamics
	state_coefficient::Function
	control_coefficient::Function
	constant::Function
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


# Type for interpolation


struct Interpolation{T <: Interpolations.BSplineInterpolation{Float64}}
	interpolator::T
	grid_steps::Array{Float64,1}
end

function eval_interpolation(x::Array{Float64,1}, i::Interpolation)

	grid_position = x ./ i.grid_steps .+ 1.
	return i.interpolator(grid_position...)
end


# Type containing all variables for Stochastic Optimal Control


mutable struct Variables
	t::Int64
	state::Union{Array{Float64,1}, Nothing}
	control::Union{Array{Float64,1}, Nothing}
	noise::Union{RandomVariable, Nothing}
end

Variables(t::Int64, rv::RandomVariable) = Variables(t, nothing, nothing, rv)
