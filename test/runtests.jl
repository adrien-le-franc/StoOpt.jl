# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, JLD, Test

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
        	noise = NNoise(w, pw)
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
        	noise = NNoise(w, pw)
        	for (val, proba) in StoOpt.iterator(noise, 3)
        		if (val, proba) == ([6.0, 12.0], 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_kmeans()
            data = rand(5, 100)
            noise = NNoise(data, 3)
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
    noises = NNoise(data["w"], data["pw"])
    states = Grid(0:0.1:1, enumerate=true)
    controls = Grid(-1:0.1:1)
    horizon = 96
    price = ones(96)*0.1
    price[28:84] .+= 0.05
    cost_parameters = Dict("buy"=>price)
    dynamics_parameters = Dict("charge"=>0.95, "discharge"=>0.95)

    function dynamics(m::SDP, t::Int64, x::Array{Float64}, u::Array{Float64}, w::Array{Float64})
        return x + (m.dynamics_parameters["charge"]*max.(0,u) 
            - max.(0,-u)/m.dynamics_parameters["discharge"])
    end

    function cost(m::SDP, t::Int64, x::Array{Float64}, u::Array{Float64}, w::Array{Float64})
        return (m.cost_parameters["buy"][t]*max.(0, u+w))[1]
    end

    @testset "SDP" begin
            
            sdp = SDP(states, controls, noises, cost_parameters, dynamics_parameters, horizon)
            value_functions = compute_value_functions(sdp, cost, dynamics)

            @test value_functions[horizon] == zeros(size(states))

            println(value_functions[1])
            println(value_functions[horizon])

            @test all(value_functions[1] .> value_functions[horizon])


    end

end

