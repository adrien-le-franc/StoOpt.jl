# developed with Julia 1.1.1
#
# models for stochastic optimal control


abstract type AbstractModel end
abstract type DynamicProgrammingModel <: AbstractModel end
abstract type SdpModel <: DynamicProgrammingModel end
abstract type AdpModel <: DynamicProgrammingModel end
abstract type SddpModel <: AdpModel end
abstract type RollingHorizonModel <: AbstractModel end


struct DummyModel <: AbstractModel end


mutable struct SDP <: SdpModel

	states::Grid
	controls::Grid
	noises::Union{Noise, Nothing}
	cost_parameters::Dict{String,Any}
	dynamics_parameters::Dict{String,Any}
	horizon::Int64

end


struct SDDP <: SddpModel
end


struct MPC <: RollingHorizonModel

	cost_parameters::Dict{String,Any}
	dynamics_parameters::Dict{String,Any}
	horizon::Int64

end