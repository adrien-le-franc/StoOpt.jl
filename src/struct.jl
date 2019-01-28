# developed with Julia 1.0.3
#
# generic struct items for EMS problems 

#using Clustering

## Grid ##

struct Grid
	"""discretized space grid"""
	states::Array{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 1}
end

Base.size(g::Grid) = Tuple([length(g.states[i]) for i in 1:length(g.states)])
Base.length(g::Grid) = length(size(g))
Base.getindex(g::Grid, i::Int) = g.states[i]

function grid_steps(g::Grid)
	"""grid steps, assuming a regular space grid: return type Tuple"""
	dimension = length(size(g))
	grid_steps = [g.states[i][2] - g.states[i][1] for i in 1:dimension]
	#Tuple(grid_steps)
end

function Grid(states::Vararg{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}})

	Grid(collect(states))

end

function Grid(xmin::Array{Float64}, dx::Array{Float64}, xmax::Array{Float64})

	if !(size(xmin) == size(dx) == size(xmax))
			error("Grid: xmin $(size(xmin)) dx $(size(dx)) xmax $(size(xmax)) do not match")
	end

	dimension = length(xmin)
	states = [xmin[i]:dx[i]:xmax[i] for i in 1:dimension]
	
	Grid(states)

end

function run(input::Grid; enumerate=false)


	if !enumerate
		return Iterators.product(input.states...)
	else
		grid_size = size(input)
		indices = Iterators.product([1:i for i in grid_size]...)
		return zip(Iterators.product(input.states...), indices)
	end

end

## Noise ## 
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

#function Noise(w::Array{Float64, 2}, pw::Array{Float64, 2}) 
#
#	Noise(w, pw)
#
#end

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
