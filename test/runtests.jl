# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, JLD, Test
using Distributed
const SO = StoOpt

current_directory = @__DIR__


@testset "StoOpt" begin

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

            value_functions = SO.ArrayValueFunctions(horizon+1, 11)
            interpolator = SO.Interpolator(horizon, states, value_functions)
            variables = SO.Variables(horizon, [0.0], [-0.1], SO.RandomVariable(noises, horizon))

            @test SO.compute_expected_realization(sdp, variables, interpolator) == Inf

            @test isapprox(SO.compute_cost_to_go(sdp, variables, interpolator),
                0.048178640538937376)

            value_functions = SO.ArrayValueFunctions(sdp.horizon, size(sdp.states)...)
            pool = CachingPool(workers())
            SO.parallel_fill_value_function!(value_functions, variables.t, sdp, 
                interpolator, pool)

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