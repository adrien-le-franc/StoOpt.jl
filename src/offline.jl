# developed with Julia 1.0.3
#
# offline step for Stochastic Dynamic Programming 


function compute_value_functions(sdp::SDP, cost::Function, dynamics::Function)

	value_functions = ArrayValueFunctions((size(sdp.states).., sdp.horizon))
	state_iterator = run(sdp.states, enumerate=true)
	control_iterator = run(sdp.controls)

	for t in sdp.horizon-1:-1:1

		noise_iterator = run(sdp.noises, t+1)

		for (state, index) in state_iterator

			state = collect(state)


			for (noise, probability) in noise_iterator

				noise = collect(noise)

	end

end