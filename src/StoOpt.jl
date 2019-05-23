module StoOpt

using ProgressMeter
using Interpolations, StatsBase, Clustering
using EllipsisNotation

include("struct.jl")

export ValueFunctions, ArrayValueFunctions
export Grid, Container, NNoise, Noise, Price

include("models.jl")

export AbstractModel, DynamicProgrammingModel
export DummyModel
export SDP, SDDP, MPC

include("utils.jl")

export admissible_state

include("sdp.jl")

export compute_value_functions
export compute_mean_risk_value_functions
export compute_online_policy
export compute_online_trajectory

include("offline.jl")

export compute_value_functions

end # module
