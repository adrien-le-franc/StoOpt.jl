module StoOpt

using Interpolations, StatsBase, Clustering, JuMP
using EllipsisNotation

import JuMP
using JuMP: Model, with_optimizer, @variable, @constraint, @expression, @objective

include("struct.jl")

export ValueFunctions, ArrayValueFunctions
export Grid, Noises, RandomVariable

include("models.jl")

export AbstractModel, DynamicProgrammingModel, SdpModel, RollingHorizonModel
export SDP, SDDP

include("utils.jl")

export admissible_state

include("offline.jl")

export compute_value_functions

include("online.jl")

export compute_control

end # module
