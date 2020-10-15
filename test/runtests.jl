# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, JLD, Test
using Distributed
const SO = StoOpt

current_directory = @__DIR__


@testset "StoOpt" begin

    @testset "struct" begin 
        
        function test_bounds()
            bounds = SO.Bounds(2, [10., 2.], [7., -1.])
            upper, _ = bounds[2]
            return upper == [10., 2.]
        end

        function test_scalar_bounds()
            bounds = SO.Bounds(2, 2., 1.)
            upper, _ = bounds[2]
            return upper == [2.]
        end

        function test_states_1()
            states = SO.States(2, 1:5., 1:5.)
            upper, lower = states.bounds[2]
            return (upper == [5., 5.] && lower == [1., 1.])
        end

        function test_states_2()
            states = SO.States(2, 1:5., 1:5.)
            x = [state for (state, index) in states.iterator]
            return x[end] == (5., 5.)
        end

        function test_controls()
            controls = SO.Controls(2, 1:2., 1:2.)
            x = [i for i in controls[2]]
            return x[1] == (1., 1.)
        end

        function test_control_bounds()
            bounds = SO.Bounds(2, [10., 2.], [7., -1.])
            controls = SO.Controls(bounds, 1:7., 1:5.)
            x = [i for i in controls[2]]
            return x[end] == (7., 2.)
        end

        function test_noise_iterator_1d()
        	w = reshape(collect(1:6.0), 3, 2)
        	pw = ones(3, 2)*0.5
        	noises = SO.Noises(w, pw)
            noise = SO.RandomVariable(noises, 3)
        	for (val, proba) in SO.iterator(noise) ## ???
        		if (val[1], proba) == (6.0, 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_iterator_2d()
        	w = reshape(collect(1:12.0), 3, 2, 2)
        	pw = ones(3, 2)*0.5
        	noises = SO.Noises(w, pw)
            noise = SO.RandomVariable(noises, 3)
        	for (val, proba) in SO.iterator(noise) ## ??
        		if (val, proba) == ([6.0, 12.0], 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_kmeans()
            data = rand(5, 100)
            noise = SO.Noises(data, 3)
            pw = noise.pw.data
            if size(pw)!= (5, 3)
                return nothing
            else
                for i in sum(pw, dims=2)
                    if !isapprox(i, 1.0)
                        return nothing
                    end
                end
            end
            return 1
        end

        @test test_bounds()
        @test test_states_1()
        @test test_states_2()
        @test test_scalar_bounds()
        @test test_controls()
        @test test_control_bounds()
        @test test_noise_iterator_1d() == 1
        @test test_noise_iterator_2d() == 1
        @test test_noise_kmeans() == 1

    end

    horizon = 96
    data = load(current_directory*"/data/test.jld")
    noises = SO.Noises(data["w"], data["pw"])
    states = SO.States(horizon, 0:0.1:1) 
    controls = SO.Controls(horizon, -1:0.1:1)
    price = ones(96)*0.1
    price[28:84] .+= 0.05
    cmax = 5.
    umax = 1.
    r = 0.95

    function cost(t::Int64, x::Array{Float64,1}, u::Array{Float64,1}, w::Array{Float64,1})
        u = u[1]*umax
        demand = u + w[1]
        return price[t]*max(0.,demand) 
    end

    function dynamics(t::Int64, x::Array{Float64,1}, u::Array{Float64,1}, w::Array{Float64,1})
        scale_factor = umax/cmax
        return x + (r*max.(0.,u) - max.(0.,-u)/r)*scale_factor
    end

    sdp = SO.SDP(states, controls, noises, cost, dynamics, horizon)

    @testset "SDP" begin
            
            value_functions = SO.ArrayValueFunctions(horizon+1, 11)
            interpolator = SO.Interpolator(horizon, states, value_functions)
            variables = SO.Variables(horizon, [0.0], [-0.1], SO.RandomVariable(noises, horizon))

            @test SO.compute_expected_realization(sdp, variables, interpolator) == Inf

            @test isapprox(SO.compute_cost_to_go(sdp, variables, interpolator),
                0.048178640538937376)

            value_functions = SO.ArrayValueFunctions(sdp.horizon, size(sdp.states)...)
            SO.fill_value_function!(value_functions, variables.t, sdp, interpolator)

            @test isapprox(value_functions[horizon][11], 0.0012163218646055842,)

            t = @elapsed value_functions = SO.compute_value_functions(sdp)

            @test t < 2.
            @test value_functions[horizon+1] == zeros(size(states))
            @test all(value_functions[1] .> value_functions[horizon])
            @test value_functions[1][1] > value_functions[1][end]
            @test SO.compute_control(sdp, 1, [0.0], SO.RandomVariable(noises, 1), 
                value_functions) == [0.0]

            sdp.final_cost = f(x::Array{Float64,1}) = 0.02*x[1]
            t = @elapsed value_functions = SO.compute_value_functions(sdp)

            @test t < 2.
            @test value_functions[horizon+1] == 0.02*collect(states.axis...)
    end

    SO.set_processors(2)

    @testset "parallel SDP" begin
            
            @everywhere using StoOpt

            value_functions = SO.SharedArrayValueFunctions(horizon+1, 11)
            interpolator = SO.Interpolator(horizon, states, value_functions)
            variables = SO.Variables(horizon, [0.0], [-0.1], SO.RandomVariable(noises, horizon))

            @test SO.compute_expected_realization(sdp, variables, interpolator) == Inf

            @test isapprox(SO.compute_cost_to_go(sdp, variables, interpolator),
                0.048178640538937376)

            value_functions = SO.SharedArrayValueFunctions(sdp.horizon, size(sdp.states)...)
            SO.parallel_fill_value_function!(value_functions, variables.t, sdp, interpolator)

            @test isapprox(value_functions[horizon][11], 0.0012163218646055842,)

            t = @elapsed value_functions = SO.parallel_compute_value_functions(sdp)

            @test t < 2.
            @test isapprox(value_functions[horizon+1], 0.02*collect(states.axis...))
            @test all(value_functions[1] .> value_functions[horizon])
            @test value_functions[1][1] > value_functions[1][end]
            @test SO.compute_control(sdp, 1, [0.0], SO.RandomVariable(noises, 1), 
                value_functions) == [0.0]

    end

    SO.set_processors(1)
    
end