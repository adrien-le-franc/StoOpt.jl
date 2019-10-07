# developed with Julia 1.1.1
#
# models for stochastic optimal control


abstract type AbstractModel end
abstract type DynamicProgrammingModel <: AbstractModel end
abstract type SdpModel <: DynamicProgrammingModel end
abstract type AdpModel <: DynamicProgrammingModel end
abstract type SddpModel <: AdpModel end
abstract type RollingHorizonModel <: AbstractModel end


mutable struct SDP <: SdpModel

	states::Grid
	controls::Grid
	noises::Union{Noises, Nothing}
	cost::Function
	dynamics::Function
	horizon::Int64

end

