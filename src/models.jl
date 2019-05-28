# developed with Julia 1.1.1
#
# models for stochastic optimal control


abstract type AbstractModel end
abstract type DynamicProgrammingModel <: AbstractModel end


struct DummyModel <: AbstractModel end


mutable struct SDP <: DynamicProgrammingModel

	states::Grid
	controls::Grid
	noises::Union{Noise, Nothing}
	cost_parameters::Dict{String,Any}
	dynamics_parameters::Dict{String,Any}
	horizon::Int64

end


#struct SDPaR <: SDP
#end
# Q: type for "stagewise dependence" models ?

struct SDDP <: DynamicProgrammingModel

end


struct MPC <: AbstractModel

end