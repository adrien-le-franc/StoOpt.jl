module StoOpt

using ProgressMeter
using Interpolations, StatsBase, Clustering
using EllipsisNotation

include("struct.jl")

export ValueFunctions, ArrayValueFunctions
export Grid, Noise

include("models.jl")

export AbstractModel, DynamicProgrammingModel
export DummyModel
export SDP, SDDP, MPC

include("utils.jl")

export admissible_state

include("offline.jl")

export compute_value_functions

end # module
