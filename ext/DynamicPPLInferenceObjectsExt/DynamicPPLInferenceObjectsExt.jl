module DynamicPPLInferenceObjectsExt

using AbstractPPL: AbstractPPL
using DimensionalData: DimensionalData, Dimensions, LookupArrays
using DynamicPPL: DynamicPPL
using InferenceObjects: InferenceObjects
using Random: Random
using StatsBase: StatsBase

include("utils.jl")
include("varinfo.jl")
include("condition.jl")
include("generated_quantities.jl")
include("predict.jl")
include("pointwise_loglikelihoods.jl")

end
