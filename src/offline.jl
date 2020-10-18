# developed with Julia 1.4.2
#
# offline step for Stochastic Dynamic Programming 


function compute_expected_realization(sdp::SdpModel, variables::Variables, 
	interpolator::Interpolator)

	expected_realization = 0.

	for (noise, probability) in iterator(variables.noise) # à changer

		noise = collect(noise)
		next_state = sdp.dynamics(variables.t, variables.state, variables.control, noise)

		if !state_in_bounds(variables.t, next_state, sdp)
			return Inf # à tester !!
		end

		next_cost_to_go = interpolator.value(next_state...)
		realization = sdp.cost(variables.t, variables.state, 
			variables.control, noise) + next_cost_to_go

		expected_realization += realization*probability

	end

	return expected_realization

end 

function compute_cost_to_go(sdp::SdpModel, variables::Variables, interpolator::Interpolator)

	cost_to_go = Inf

	for control in sdp.controls[variables.t]

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, variables, interpolator)
		cost_to_go = min(cost_to_go, realization)

	end

	return cost_to_go

end

function fill_value_function!(value_functions::ArrayValueFunctions, t::Int64, sdp::SdpModel, 
	interpolator::Interpolator)

	variables = Variables(t, RandomVariable(sdp.noises, t))

	for (state, index) in sdp.states.iterator

		variables.state = collect(state)
		index = (variables.t, index...)
		value_functions[index...] = compute_cost_to_go(sdp, variables, interpolator)

	end

	return nothing

end

function initialize_value_functions(sdp::SdpModel)

	value_functions = ArrayValueFunctions(sdp.horizon+1, size(sdp.states)...)

	if !isnothing(sdp.final_cost)

		final_values = zeros(size(sdp.states))
		for (state, index) in sdp.states.iterator
			state = collect(state)
			final_values[index...] = sdp.final_cost(state)
		end
		value_functions[sdp.horizon+1] = final_values
		
	end

	return value_functions

end

function compute_value_functions(sdp::SdpModel)

	value_functions = initialize_value_functions(sdp)
	interpolator = Interpolator(sdp.horizon+1, sdp.states, value_functions)

	for t in sdp.horizon:-1:1

		fill_value_function!(value_functions, t, sdp, interpolator)
		update_interpolator!(interpolator, t, sdp.states, value_functions)

	end

	return value_functions

end