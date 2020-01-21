module StoOpt

using Interpolations, StatsBase, Clustering
using EllipsisNotation
using JuMP, Clp # ??
using MathOptInterface
const MOI = MathOptInterface

include("struct.jl")

export ValueFunctions, ArrayValueFunctions
export Grid, Noises, RandomVariable

include("models.jl")
include("utils.jl")
include("offline.jl")

export compute_value_functions

include("online.jl")

export compute_control

end # module
