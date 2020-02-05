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
	dynamics::LinearDynamics
	horizon::Int64

end

function SDDP(state_bounds::Bounds, control_bounds::Bounds, 
	noises::Noises, cost::PolyhedralCost, dynamics::LinearDynamics, horizon::Int64)

	coefficient = dynamics.state_coefficient(1, noises.w[1][1, :])
	state_dimension = state_bounds.n_variables
	if size(coefficient) != (state_dimension, state_dimension)
		error("state_coefficient should return Array{Float64,2} of size $((state_dimension, 
			state_dimension)) but returned object of size $(size(coefficient))")
	end

	coefficient = dynamics.control_coefficient(1, noises.w[1][1, :])
	control_dimension = control_bounds.n_variables
	if size(coefficient) != (state_dimension, control_dimension)
		error("control_coefficient should return Array{Float64,2} of size $((state_dimension, 
			control_dimension)) but returned object of size $(size(coefficient))")
	end

	coefficient = dynamics.constant(1, noises.w[1][1, :])
	if size(coefficient) != (state_dimension, 1)
		error("state_coefficient should return Array{Float64,2} of size $((state_dimension, 
			1)) but returned object of size $(size(coefficient))")
	end
	
    return SDDP(Model(), state_bounds, control_bounds, noises, cost, dynamics, horizon)

end