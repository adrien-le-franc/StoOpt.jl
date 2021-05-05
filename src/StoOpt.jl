module StoOpt

using ControlVariables
const Variables = ControlVariables.Variables
const RandomVariable = ControlVariables.RandomVariable
const law = ControlVariables.law

using Interpolations, StatsBase
using EllipsisNotation
using Distributed

include("struct.jl")
include("models.jl")
include("utils.jl")
include("offline.jl")
include("parallel_offline.jl")
include("online.jl")

end 
