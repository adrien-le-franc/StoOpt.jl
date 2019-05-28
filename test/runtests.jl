# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, JLD, Test
using Interpolations

current_directory = @__DIR__


@testset "StoOpt" begin

    @testset "struct" begin 
        
        function test_grid_iterator()
        	g = Grid(1:3.0, 10:12.0, enumerate=true)
        	for (val, index) in g.iterator
        		if index == (2, 3) && val == (2.0, 12.0)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_iterator_1d()
        	w = reshape(collect(1:6.0), 3, 2)
        	pw = ones(3, 2)*0.5
        	noise = Noise(w, pw)
        	for (val, proba) in StoOpt.iterator(noise, 3)
        		if (val[1], proba) == (6.0, 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_iterator_2d()
        	w = reshape(collect(1:12.0), 3, 2, 2)
        	pw = ones(3, 2)*0.5
        	noise = Noise(w, pw)
        	for (val, proba) in StoOpt.iterator(noise, 3)
        		if (val, proba) == ([6.0, 12.0], 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_kmeans()
            data = rand(5, 100)
            noise = Noise(data, 3)
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

        @test test_grid_iterator() == 1
        @test test_noise_iterator_1d() == 1
        @test test_noise_iterator_2d() == 1
        @test test_noise_kmeans() == 1

    end

    data = load(current_directory*"/data/test.jld")
    noises = Noise(data["w"], data["pw"])
    states = Grid(0:0.1:1, enumerate=true)
    controls = Grid(-1:0.1:1)
    horizon = 96
    price = ones(96)*0.1
    price[28:84] .+= 0.05
    cost_parameters = Dict("buy"=>price, "umax"=>1.)
    dynamics_parameters = Dict("charge"=>0.95, "discharge"=>0.95, "cmax"=>5., "umax"=>1.)

    function dynamics(m::SDP, t::Int64, x::Array{Float64}, u::Array{Float64}, w::Array{Float64})
        normalize = m.dynamics_parameters["umax"]/m.dynamics_parameters["cmax"]
        return x + (m.dynamics_parameters["charge"]*max.(0,u) 
            - max.(0,-u)/m.dynamics_parameters["discharge"])*normalize
    end

    function cost(m::SDP, t::Int64, x::Array{Float64}, u::Array{Float64}, w::Array{Float64})
        u = u*m.cost_parameters["umax"]
        return (m.cost_parameters["buy"][t]*max.(0, u+w))[1]
    end

    @testset "SDP" begin
            
            sdp = SDP(states, controls, noises, cost_parameters, dynamics_parameters, horizon)

            interpolation = StoOpt.Interpolation(interpolate(zeros(11), BSpline(Linear())),
            states.steps)
            variables = StoOpt.Variables(horizon-1, [0.0], [-0.1], nothing)

            @test StoOpt.compute_expected_realization(sdp, cost, dynamics, variables, 
                interpolation) == Inf

            @test isapprox(StoOpt.compute_cost_to_go(sdp, cost, dynamics, variables, interpolation),
                0.048178640538937376)

            value_functions = StoOpt.ArrayValueFunctions((sdp.horizon, size(sdp.states)...))
            StoOpt.fill_value_function!(sdp, cost, dynamics, variables, value_functions,
                interpolation)

            @test isapprox(value_functions[horizon-1][11], 0.0012163218646055842,)

            t = @elapsed value_functions = compute_value_functions(sdp, cost, dynamics)

            @test t < 8.
            @test value_functions[horizon] == zeros(size(states))
            @test all(value_functions[1] .> value_functions[horizon])
            @test compute_control(sdp, cost, dynamics, 1, [0.0], value_functions) == [0.0]

    end

end

