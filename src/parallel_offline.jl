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

function bellman_operator(t::Int64, x::Tuple{Vararg{Float64}}, noises::RandomVariable,
	sdp::SdpModel, interpolator::Interpolator)

	variables = Variables(t, collect(x), nothing, noises)
	value = compute_cost_to_go(sdp, variables, interpolator)

	return value
end

function parallel_fill_value_function!(value_functions::ArrayValueFunctions, t::Int64, 
	sdp::SdpModel, interpolator::Interpolator, pool::CachingPool)

	noises = RandomVariable(sdp.noises, t)

	value_functions[t] = pmap(x->bellman_operator(t, x, noises, sdp, interpolator),
		pool, Iterators.product(sdp.states.axis...))

	return nothing

end

function initialize_value_functions(x::Tuple{Vararg{Float64}},
	sdp::SdpModel)

	state = collect(x)
	return sdp.final_cost(state)
	
end

function parallel_initialize_value_functions(sdp::SdpModel, pool::CachingPool)

	value_functions = ArrayValueFunctions(sdp.horizon+1, size(sdp.states)...)

	if !isnothing(sdp.final_cost)
		value_functions[sdp.horizon+1] = pmap(x->initialize_value_functions(x, sdp),
			pool, Iterators.product(sdp.states.axis...))	
	end

	return value_functions

end

function parallel_compute_value_functions(sdp::SdpModel)

	pool = CachingPool(workers())
	value_functions = parallel_initialize_value_functions(sdp, pool)
	interpolator = Interpolator(sdp.horizon+1, sdp.states, value_functions)

	for t in sdp.horizon:-1:1

		parallel_fill_value_function!(value_functions, t, sdp, interpolator, pool)
		update_interpolator!(interpolator, t, sdp.states, value_functions)

	end

	clear!(pool) # ??

	return value_functions

end

