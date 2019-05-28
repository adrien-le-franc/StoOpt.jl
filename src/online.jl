# developed with Julia 1.1.1
#
# online step for Stochastic Dynamic Programming 


# Dummy Model

function compute_control(m::DummyModel, cost::Function, dynamics::Function, 
		t::Int64, state::Array{Float64,1}, value_functions::Nothing)
	return 0.0
end

# SDP

function compute_control(sdp::SDP, cost::Function, dynamics::Function, 
	t::Int64, state::Array{Float64,1}, value_functions::ValueFunctions)
	
	variables = Variables(t, state, nothing, nothing)
	interpolation = Interpolation(interpolate(value_functions[t+1], BSpline(Linear())),
			sdp.states.steps)

	cost_to_go = Inf
	optimal_control = Inf

	for control in sdp.controls.iterator

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, cost, dynamics, variables, interpolation)

		if realization < cost_to_go
			cost_to_go = realization
			optimal_control = collect(control)
		end

	end

	return optimal_control

end

# SDDP