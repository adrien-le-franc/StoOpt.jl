# developed with Julia 1.0.3
#
# models for stochastic optimal control


abstract type AbstractModel end
abstract type DynamicProgrammingModel <: AbstractModel end


struct DummyModel <: AbstractModel end


struct SDP <: DynamicProgrammingModel

	states::Grid
	controls::Grid
	noises::Noise
	cost_parameters::Dict{String,Any}
	dynamics_parameters::Dict{String,Any}
	horizon::Int64

end


struct SDPaR <: SDP

end


struct SDDP <: DynamicProgrammingModel

end


struct MPC <: AbstractModel

end