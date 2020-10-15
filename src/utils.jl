# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


function state_in_bounds(t::Int64, state::Array{Float64,1}, sdp::SdpModel)
	upper, lower = sdp.states.bounds[t+1]	
	return all(lower .<= state .<= upper)
end
