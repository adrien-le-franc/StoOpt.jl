# developed with Julia 1.0.3
#
# functions for Stochastic Dynamic Programming 

using ProgressMeter, Interpolations

include("struct.jl")

function admissible_state(x::Array{Float64}, states::Grid)
	"""check if x is in states: return a boolean

	x > state point
	states > discretized state space

	"""

	for i in 1:length(x)

		if x[i] < states[i][1]
			return false
		elseif x[i] > states[i][end]
			return false
		end

	end

	return true

end

function compute_value_functions(train_noises::Union{Noise, Array{Noise}}, 
	controls::Grid, states::Grid, dynamics::Function, cost::Function, 
	prices::Array{Float64}, horizon::Int64; order::Int64=1)

	"""compute value functions: return Dict(1=>Array ... horizon=>Array)

	train_noise > noise training data
	controls, states > discretized control and state spaces 
	dynamics > function(x, u, w) returning next state
	cost > function(p, x, u, w) returning stagewise cost
	price > price per period
	horizon > time horizon
	order > interpolation order

	"""

	state_size = size(states)
	state_steps = grid_steps(states)
	state_iterator = run(states, enumerate=true)
	control_iterator = run(controls)

	value_function = [zeros(state_size...) for t in 1:T+1]
	expectation = 0.

	@showprogress for t in horizon:-1:1

		price = prices[t, :]
		noise_iterator = run(train_noises, t)

		interpolator = interpolate(value_function[t+1], BSpline(Linear()))
		
		for (state, index) in state_iterator

			state = collect(state)
			expectation = 0.

			for (noise, probability) in noise_iterator

				noise = collect(noise)
				p_noise = prod(probability)
				v = 10e8

				for control in control_iterator

					control = collect(control)
					next_state = dynamics(state, control, noise)

					if !admissible_state(next_state, states)
						continue
					end

					where = next_state ./ state_steps .+ 1.
					next_value_function = interpolator(where...)

					v = min(v, cost(price, state, control, noise) + next_value_function)

				end

				expectation += v*p_noise

			end

			value_function[t][index...] = expectation

		end

	end

	return value_function

end

function compute_online_policy(x, w, price, controls, cost, interpolation, xdict, value_function)

	"""compute online policy: return optimal control at state x observing w

	x > current state
	w > observed noise 
	price > price at current stage
	controls > discretized space as Tuple of ranges e.g. (1:10, 1:10)
	cost > function returning stagewise cost
	interpolation > fonction returning interpolated value function
	xdict > Dict mapping states to positions in value_function 
	value_function > multi dimensional array

	"""

	vopt = 10e8
    uopt = 0
        
    for u in product(controls...)
        
        next_state = dynamics(x, u, w)

		if !admissible_state(next_state, states)
			continue
		end

		if is_on_grid(next_state, states)
			next_value_function = value_function[t+1][xdict[next_state]]
		else
			next_value_function = interpolation(next_state, states, value_function[t+1])
		end

        v = cost(price, x, u, w) + next_value_function

        if v < vopt

            vopt = v
            uopt = u
        
        end
        
    end
    
    return uopt, vopt

end