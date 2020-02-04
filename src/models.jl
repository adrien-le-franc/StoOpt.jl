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
	
    return SDDP(Model(), state_bounds, control_bounds, noises, cost, dynamics, horizon)

end