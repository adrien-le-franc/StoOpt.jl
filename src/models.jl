# developed with Julia 1.0.3
#
# models for stochastic optimal control


abstract type AbstractModel end
abstract type DynamicProgrammingModel <: AbstractModel end


struct DummyModel <: AbstractModel end


struct SDP <: DynamicProgrammingModel

end


struct SDDP <: DynamicProgrammingModel

end


struct MPC <: AbstractModel

end