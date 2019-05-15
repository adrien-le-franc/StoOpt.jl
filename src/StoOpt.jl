module StoOpt

using ProgressMeter
using Interpolations, StatsBase, Clustering
using EllipsisNotation

include("struct.jl")
include("models.jl")
include("sdp.jl")

export ValueFunctions, Grid, Noise, Price

export AbstractModel, DynamicProgrammingModel
export DummyModel
export SDP, SDDP, MPC

export compute_value_functions
export compute_mean_risk_value_functions
export compute_online_policy
export compute_online_trajectory

end # module
