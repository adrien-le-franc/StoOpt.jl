# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


# SDP 

function compute_optimal_realization(sdp::SDP, cost::Function, dynamics::Function, 
	variables::Variables, interpolation::Interpolation)
	
	v = Inf

	for control in sdp.controls.iterator

		control = collect(control)
		next_state = dynamics(sdp, variables.t, variables.state, control, variables.noise)

		if !admissible_state(next_state, sdp.states)
			continue
		end

		next_value_function = eval_interpolation(next_state, interpolation)
		v = min(v, cost(sdp, variables.t, variables.state, control, variables.noise) + 
			next_value_function)

	end

	return v

end

function compute_cost_to_go(sdp::SDP, cost::Function, dynamics::Function, variables::Variables,
	interpolation::Interpolation)

	realizations = Float64[] 
	probabilities = Float64[]

	for (noise, probability) in iterator(sdp.noises, variables.t+1)

		variables.noise = collect(noise)
		realization = compute_optimal_realization(sdp, cost, dynamics, variables, interpolation)

		push!(realizations, realization)
		push!(probabilities, probability)

	end

	cost_to_go = realizations'*probabilities
	return cost_to_go

end

function fill_value_function!(sdp::SDP, cost::Function, dynamics::Function, variables::Variables,
	value_functions::ArrayValueFunctions, interpolation::Interpolation)

	value_function = ones(size(sdp.states))

	for (state, index) in sdp.states.iterator

		variables.state = collect(state)
		value_function[index...] = compute_cost_to_go(sdp, cost, dynamics, 
			variables, interpolation)

	end

	value_functions[variables.t] = value_function

	return nothing

end

function compute_value_functions(sdp::SDP, cost::Function, dynamics::Function)

	value_functions = ArrayValueFunctions((sdp.horizon, size(sdp.states)...))
	state_steps = steps(sdp.states)

	for t in sdp.horizon-1:-1:1

		variables = Variables(t)
		interpolation = Interpolation(interpolate(value_functions[t+1], BSpline(Linear())),
			state_steps)

		fill_value_function!(sdp, cost, dynamics, variables, value_functions, interpolation)

	end

	return value_functions

end

# SDDP 
