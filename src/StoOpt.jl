module StoOpt

using ProgressMeter, Interpolations, StatsBase, Clustering

include("struct.jl")
include("sdp.jl")

export compute_value_functions
export compute_mean_risk_value_functions
export compute_online_policy
export compute_online_trajectory

end # module
