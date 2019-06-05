# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


# SDP 

function compute_expected_realization(sdp::SdpModel, cost::Function, dynamics::Function, 
	variables::Variables, interpolation::Interpolation)

	realizations = Float64[] 
	probabilities = Float64[]
	reject_control = Float64[]

	for (noise, probability) in iterator(variables.noise)

		noise = collect(noise)
		next_state = dynamics(sdp, variables.t, variables.state, variables.control, noise)

		if !admissible_state!(next_state, sdp.states)
			push!(reject_control, probability)
		end

		next_value_function = eval_interpolation(next_state, interpolation)
		realization = cost(sdp, variables.t, variables.state, variables.control, noise) + 
			next_value_function

		push!(realizations, realization)
		push!(probabilities, probability)

	end

	if isapprox(sum(reject_control), 1.0)
		return Inf
	else
		expected_cost_to_go = realizations'*probabilities
		return expected_cost_to_go
	end

end 

function compute_cost_to_go(sdp::SdpModel, cost::Function, dynamics::Function, 
	variables::Variables, interpolation::Interpolation)

	cost_to_go = Inf

	for control in sdp.controls.iterator

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, cost, dynamics, variables, interpolation)
		cost_to_go = min(cost_to_go, realization)

	end

	return cost_to_go

end

function fill_value_function!(sdp::SdpModel, cost::Function, dynamics::Function, 
	variables::Variables, value_functions::ArrayValueFunctions, interpolation::Interpolation)

	value_function = ones(size(sdp.states))

	for (state, index) in sdp.states.iterator

		variables.state = collect(state)
		value_function[index...] = compute_cost_to_go(sdp, cost, dynamics, 
			variables, interpolation)

	end

	value_functions[variables.t] = value_function

	return nothing

end

function compute_value_functions(sdp::SdpModel, cost::Function, dynamics::Function)

	value_functions = ArrayValueFunctions((sdp.horizon+1, size(sdp.states)...))

	for t in sdp.horizon:-1:1

		variables = Variables(t, RandomVariable(sdp.noises, t))
		interpolation = Interpolation(interpolate(value_functions[t+1], BSpline(Linear())),
			sdp.states.steps)

		fill_value_function!(sdp, cost, dynamics, variables, value_functions, interpolation)

	end

	return value_functions

end

# SDDP 
