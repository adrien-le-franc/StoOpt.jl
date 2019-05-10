module StoOpt

using ProgressMeter, Interpolations, StatsBase, Clustering

include("struct.jl")
include("models.jl")
include("sdp.jl")

export Grid, Noise, Price

export AbstractModel, DynamicProgrammingModel
export SDP

export compute_value_functions
export compute_mean_risk_value_functions
export compute_online_policy
export compute_online_trajectory

end # module
