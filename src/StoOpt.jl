module StoOpt

using Interpolations, StatsBase, Clustering
using EllipsisNotation
using Distributed, SharedArrays

include("struct.jl")
include("models.jl")
include("utils.jl")
include("offline.jl")
include("parallel_offline.jl")
include("online.jl")

end # module
