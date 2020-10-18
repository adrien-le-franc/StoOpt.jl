# developed with Julia 1.1.1
#
# online step for Stochastic Dynamic Programming 


function compute_control(sdp::SdpModel, t::Int64, state::Array{Float64,1}, 
	noise::RandomVariable, value_functions::ValueFunctions)
	
	variables = Variables(t, state, nothing, noise)
	interpolator = Interpolator(t+1, sdp.states, value_functions)

	cost_to_go = Inf
	optimal_control = Inf

	for control in sdp.controls[t]

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, variables, interpolator)

		if realization < cost_to_go
			cost_to_go = realization
			optimal_control = collect(control)
		end

	end

	return optimal_control

end