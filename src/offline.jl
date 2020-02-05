# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


# SDP 

function compute_expected_realization(sdp::SdpModel, variables::Variables, 
    interpolation::Interpolation)

    realizations = Float64[] 
    probabilities = Float64[]
    reject_control = Float64[]

    for (noise, probability) in iterator(variables.noise)

        noise = collect(noise)
        next_state = sdp.dynamics(variables.t, variables.state, variables.control, noise)

        if !admissible_state!(next_state, sdp.states)
            push!(reject_control, probability)
        end

        next_value_function = eval_interpolation(next_state, interpolation)
        realization = sdp.cost(variables.t, variables.state, variables.control, noise) + 
            next_value_function

        push!(realizations, realization)
        push!(probabilities, probability)

    end

    if isapprox(sum(reject_control), 1.0)
        return Inf
    else
        expected_cost_to_go = realizations'*probabilities
        return expected_cost_to_go
    end

end 

function compute_cost_to_go(sdp::SdpModel, variables::Variables, interpolation::Interpolation)

    cost_to_go = Inf

    for control in sdp.controls.iterator

        variables.control = collect(control)
        realization = compute_expected_realization(sdp, variables, interpolation)
        cost_to_go = min(cost_to_go, realization)

    end

    return cost_to_go

end

function fill_value_function!(sdp::SdpModel, variables::Variables, 
    value_functions::ArrayValueFunctions, interpolation::Interpolation)

    value_function = ones(size(sdp.states))

    for (state, index) in sdp.states.iterator

        variables.state = collect(state)
        value_function[index...] = compute_cost_to_go(sdp, variables, interpolation)

    end

    value_functions[variables.t] = value_function

    return nothing

end

function compute_value_functions(sdp::SdpModel)

    value_functions = ArrayValueFunctions((sdp.horizon+1, size(sdp.states)...))

    for t in sdp.horizon:-1:1

        variables = Variables(t, RandomVariable(sdp.noises, t))
        interpolation = Interpolation(interpolate(value_functions[t+1], BSpline(Linear())),
            sdp.states.steps)

        fill_value_function!(sdp, variables, value_functions, interpolation)

    end

    return value_functions

end


# SDDP 


function initialize_model!(sddp::SDDP)
    # generalize to other solvers
    #model = Model(with_optimizer(CPLEX.Optimizer))
    #MOI.set(model, MOI.RawParameter("CPX_PARAM_SCRIND"), 0)
    sddp.model = Model(with_optimizer(Clp.Optimizer, LogLevel=0))
    return nothing
end

function initialize_variables!(sddp::SDDP)

    @variable(sddp.model, state[1:sddp.state_bounds.n_variables])
    @constraint(sddp.model, 
        sddp.state_bounds.lower_bounds .<= state .<= sddp.state_bounds.upper_bounds)

    @variable(sddp.model, control[1:sddp.control_bounds.n_variables])
    @constraint(sddp.model, 
        sddp.control_bounds.lower_bounds .<= control .<= sddp.control_bounds.upper_bounds)

    return nothing

end

function initialize_polyhedral_cost!(sddp::SDDP)

    model = sddp.model

    @variable(model, auxiliary_cost[1:sddp.noises.w.cardinal])
    @constraint(model, cost_constraints[i=1:sddp.cost.n_cuts, j=1:sddp.noises.w.cardinal], 
        0*sum(model[:state]) + 0*sum(model[:control]) - auxiliary_cost[j] <= 0.)

    return nothing

end

function initialize_polyhedral_value_functions!(sddp::SDDP, max_cuts::Int64)

    model = sddp.model

    @variable(model, auxiliary_value_function[1:sddp.noises.w.cardinal])
    @constraint(model, value_function_constraints[i=1:max_cuts, j=1:sddp.noises.w.cardinal],
        0*sum(model[:state]) + 0*sum(model[:control]) - 0*auxiliary_value_function[j] <= 0.)

    return nothing

end

function initialize_sddp!(sddp::SDDP; max_iterations::Int64=10)
    
    initialize_model!(sddp)
    initialize_variables!(sddp)
    initialize_polyhedral_cost!(sddp)
    initialize_polyhedral_value_functions!(sddp, max_iterations)

    return nothing

end

function update_polyhedral_cost!(sddp::SDDP, t::Int64)

    for j in 1:sddp.noises.w.cardinal

        sddp.cost.update_cost!(sddp.model[:cost_constraints][:, j], t, 
            sddp.model[:state], sddp.model[:control], sddp.noises.w[t][j, :])
    
    end

    return nothing

end

function update_value_function_cut!(sddp::SDDP, value_functions::CutsValueFunctions, 
    t::Int64, i::Int64, j::Int64)
    
    model = sddp.model
    alpha = value_functions.functions[t+1].alpha[i]
    beta = value_functions.functions[t+1].beta[i]

    state_coefficients = sddp.dynamics.state_coefficient(t, sddp.noises.w[t][j, :])'*alpha
    set_normalized_coefficient.(model[:value_function_constraints][i, j], model[:state], 
        state_coefficients)

    control_coefficients = sddp.dynamics.control_coefficient(t, sddp.noises.w[t][j, :])'*alpha
    set_normalized_coefficient.(model[:value_function_constraints][i, j], model[:control], 
        control_coefficients)

    constant = (sddp.dynamics.constant(t, sddp.noises.w[t][j, :])'*alpha)[1] + beta
    set_normalized_rhs(model[:value_function_constraints][i, j], -constant)

    set_normalized_coefficient(model[:value_function_constraints][i, j], 
        model[:auxiliary_value_function][j], -1.)

    return nothing

end

function update_polyhedral_value_functions!(sddp::SDDP, value_functions::CutsValueFunctions, 
    t::Int64, k::Int64)
    
    for j in 1:sddp.noises.w.cardinal
       for i in 1:k

            update_value_function_cut!(sddp, value_functions, t, i, j)

       end 
    end

    return nothing

end

function forward_pass(sddp::SDDP, value_functions::CutsValueFunctions, k::Int64)

    if k == 1
        trajectory = 0
    else
        trajectory = 0
    end

    return trajectory

end

function backward_pass()
end

function compute_value_functions(sddp::SddpModel; max_iterations::Int64=10)

    initialize_sddp!(sddp)
    value_functions = CutsValueFunctions(sddp.horizon+1)
    
    for k in 1:max_iterations

        trajectory = forward_pass(sddp, value_functions, k)
        stopping_criterion = backward_pass!(sddp, trajectory, value_functions)

    end

    return value_functions

end
