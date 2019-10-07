# developed with Julia 1.1.1
#
# online step for Stochastic Dynamic Programming 


# SDP

function compute_control(sdp::SdpModel, t::Int64, state::Array{Float64,1}, 
	noise::RandomVariable, value_functions::ValueFunctions)
	
	variables = Variables(t, state, nothing, noise)
	interpolation = Interpolation(interpolate(value_functions[t+1], BSpline(Linear())),
			sdp.states.steps)

	cost_to_go = Inf
	optimal_control = Inf

	for control in sdp.controls.iterator

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, variables, interpolation)

		if realization < cost_to_go
			cost_to_go = realization
			optimal_control = collect(control)
		end

	end

	return optimal_control

end

# SDDP