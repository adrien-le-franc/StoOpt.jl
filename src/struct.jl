# developed with Julia 1.1.1
#
# generic struct for Stochastic Optimization problems 


# Types for value functions


abstract type ValueFunctions end


struct ArrayValueFunctions <: ValueFunctions
	functions::Array{Float64}
end

Base.getindex(vf::ArrayValueFunctions, t::Int64) = vf.functions[t, ..]
Base.:(==)(vf1::ArrayValueFunctions, vf2::ArrayValueFunctions) = (vf1.functions == vf2.functions)
ArrayValueFunctions(t::Tuple{Vararg{Int64}}) = ArrayValueFunctions(zeros(t))


# Type for discretized spaces 


struct Grid{T <: StepRangeLen{Float64}}
	states::Array{T}
end

Base.getindex(g::Grid, i::Int) = g.states[i]
Base.size(g::Grid) = Tuple([length(g[i]) for i in 1:length(g.states)])

function Grid(states::Vararg{T}) where T <: StepRangeLen{Float64}
	Grid(collect(states))
end

function run(g::Grid{T}; enumerate=false) where T <: StepRangeLen{Float64}

	if !enumerate
		return Iterators.product(g.states...)
	else
		grid_size = size(g)
		indices = Iterators.product([1:i for i in grid_size]...)
		return zip(Iterators.product(g.states...), indices)
	end

end

#function grid_steps(g::Grid)
#	"""grid steps, assuming a regular space grid: return type Tuple"""
#	dimension = length(size(g))
#	grid_steps = [g.states[i][2] - g.states[i][1] for i in 1:dimension]
#end


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
run(tp::TimeProcess, t::Int64) = eachrow(tp[t])

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
run(p::Probability, t::Int64) = Iterators.Stateful(p[t])


struct NNoise
	w::TimeProcess
	pw::Probability

	function NNoise(w::TimeProcess, pw::Probability)
		h_w, c_w, _ = size(w)
		h_pw, c_pw = size(pw)
		if (h_w, c_w) != (h_pw, c_pw)
			error("Noise: noise size $(size(w)) not compatible with probabilities size $(size(pw))")
		end
		new(w, pw)
	end
end

run(n::NNoise, t::Int64) = zip(run(n.w, t), run(n.pw, t))
NNoise(w::Array{Float64}, pw::Array{Float64}) = NNoise(TimeProcess(w), Probability(pw))

function NNoise(data::Array{Float64,2}, k::Int64) 

	"""dicretize noise space to k values using Kmeans: return type Noise
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
		pw[t, :] = kmeans_w.cweights / n_data
	end

	return NNoise(w, pw)

end


























struct Noise
	"""discretized noise space with probabilities"""

	w::Array{Float64, 2}
	pw::Array{Float64, 2}

	function Noise(w::Array{Float64, 2}, pw::Array{Float64, 2})

		if size(w) != size(pw)
			error("Noise: noise size $(size(w)) not equal to probabilities size $(size(pw))")
		end
		new(w, pw)

	end

end


function Noise(data::Array{Float64, 2}, k::Int64) 

	"""dicretize noise space to k values using Kmeans: return type Noise
	data > time series data of dimension (horizon, n_data)
	k > Kmeans parameter

	"""

	horizon, n_data = size(data)
	w = zeros(horizon, k)
	pw = zeros(horizon, k)

	for t in 1:1:horizon
		w_t = reshape(data[t, :], (1, :))
		kmeans_w = kmeans(w_t, k)
		w[t, :] = kmeans_w.centers
		pw[t, :] = kmeans_w.cweights / n_data
	end

	return Noise(w, pw)

end

function run(input::Union{Noise, Array{Noise}}, i::Int64) 

	if input isa Noise

		return Iterators.zip(input.w[i, :], input.pw[i, :])

	else

		w = [input[1].w[i, :]]
		p = [input[1].pw[i, :]]

		for j in 2:length(input)

			push!(w, input[j].w[i, :])
			push!(p, input[j].pw[i, :])

		end

		return Iterators.zip(Iterators.product(w...), Iterators.product(p...))

	end

end








struct Price
	"""purchase and sale prices"""
	buy::Array{Float64, 1}
	sell::Array{Float64, 1}

	function Price(buy::Array{Float64, 1}, sell::Array{Float64, 1})

		if size(buy) != size(sell)
			error("Price: buy size $(size(buy)) not equal to sell size $(size(sell))")
		end
		new(buy, sell)

	end

end

function Price(buy::Array{Float64, 1})
	Price(buy, zeros(size(buy)))
end

function Price()
	Price([0.])
end

Base.length(p::Price) = length(p.buy)
Base.getindex(p::Price, i::Int) = [p.buy[i], p.sell[i]]