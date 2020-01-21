# developed with Julia 1.1.1
#
# models for stochastic optimal control


abstract type AbstractStochasticModel end
abstract type SdpModel <: AbstractStochasticModel end
abstract type SddpModel <: AbstractStochasticModel end
#abstract type DynamicProgrammingModel <: AbstractStochasticModel end
#abstract type SddpModel <: AdpModel end
#abstract type AdpModel <: DynamicProgrammingModel end
#abstract type RollingHorizonModel <: AbstractModel end


mutable struct SDP <: SdpModel

	states::Grid
	controls::Grid
	noises::Union{Noises, Nothing}
	cost::Union{Function, Nothing}
	dynamics::Union{Function, Nothing}
	horizon::Int64

end


mutable struct SDDP <: SddpModel

	model::Model
	state_bounds::Bounds
	control_bounds::Bounds
	noises::Noises
	cost::PolyhedralCost
	dynamics::Function # specific Type ?
	horizon::Int64

end

function SDDP(state_bounds::Bounds, control_bounds::Bounds, 
	noises::Noises, cost::PolyhedralCost, dynamics::Function, horizon::Int64)
	
	#model = Model(with_optimizer(CPLEX.Optimizer))
    #MOI.set(model, MOI.RawParameter("CPX_PARAM_SCRIND"), 0) 

    model = Model(with_optimizer(Clp.Optimizer, LogLevel=0))

    @variable(model, state[1:state_bounds.n_variables])
    @constraint(model, state_bounds.lower_bounds .<= state .<= state_bounds.upper_bounds)

    @variable(model, control[1:control_bounds.n_variables])
    @constraint(model, control_bounds.lower_bounds .<= control .<= control_bounds.upper_bounds)

    @variable(model, auxiliary_cost)
    @constraint(model, cost_constraints[1:cost.n_cuts], 
    	sum(state) + sum(control) + auxiliary_cost <= 0.) # !! n_cuts * K !!

    cost.update_cost!(model, 1, noises)

    return SDDP(model, state_bounds, control_bounds, noises, cost, dynamics, horizon)

end