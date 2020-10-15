# developed with Julia 1.4.2
#
# offline step for Stochastic Dynamic Programming 


function set_processors(n::Integer = Sys.CPU_THREADS; kw...)
    if n < 1
        error("number of workers must be greater than 0")
    elseif n == 1 && workers() != [1]
        rmprocs(workers())
    elseif n > nworkers()
        p = addprocs(n - (nprocs() == 1 ? 0 : nworkers()); kw...)
    elseif n < nworkers()
        rmprocs(workers()[n + 1:end])
    end
    return workers()
end

function parallel_fill_value_function!(value_functions::SharedArrayValueFunctions, t::Int64, 
	sdp::SdpModel, interpolator::Interpolator)

	variables = Variables(t, RandomVariable(sdp.noises, t))

	@sync @distributed for (state, index) in collect(sdp.states.iterator)

		variables.state = collect(state)
		index = (variables.t, index...)
		value_functions[index...] = compute_cost_to_go(sdp, variables, interpolator)

	end

	return nothing

end

function initialize_shared_value_functions(sdp::SdpModel)

	value_functions = SharedArrayValueFunctions(sdp.horizon+1, size(sdp.states)...)

	if !isnothing(sdp.final_cost)
		@sync @distributed for (state, index) in collect(sdp.states.iterator)
			state = collect(state)
			index = (sdp.horizon+1, index...)
			value_functions[index...] = sdp.final_cost(state)
		end	
	end

	return value_functions

end

function parallel_compute_value_functions(sdp::SdpModel)

	value_functions = initialize_shared_value_functions(sdp)
	interpolator = Interpolator(sdp.horizon, sdp.states, value_functions)

	for t in sdp.horizon:-1:1

		parallel_fill_value_function!(value_functions, t, sdp, interpolator)
		update_interpolator!(interpolator, t, sdp.states, value_functions)

	end

	return ArrayValueFunctions(value_functions)

end