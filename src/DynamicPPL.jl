module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedCollections, OrderedDict
using Printf: Printf

using AbstractMCMC: AbstractMCMC
using ADTypes: ADTypes
using BangBang: BangBang, push!!, empty!!, setindex!!
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Accessors: Accessors
using LogDensityProblems: LogDensityProblems

using LinearAlgebra: LinearAlgebra, Cholesky

using DocStringExtensions

using Random: Random

# For extending
import AbstractPPL: predict, hasvalue, getvalue

# TODO: Remove these when it's possible.
import Bijectors: link, invlink

import Base:
    Symbol,
    ==,
    hash,
    getindex,
    setindex!,
    push!,
    show,
    isempty,
    empty!,
    getproperty,
    setproperty!,
    keys,
    haskey

# VarInfo
export AbstractVarInfo,
    VarInfo,
    VarNamedTuple,
    @vnt,
    map_pairs!!,
    map_values!!,
    apply!!,
    densify!!,
    push!!,
    empty!!,
    subset,
    getlogp,
    setlogp!!,
    setlogprior!!,
    setlogjac!!,
    setloglikelihood!!,
    acclogp,
    acclogp!!,
    acclogjac!!,
    acclogprior!!,
    accloglikelihood!!,
    is_transformed,
    set_transformed!!,
    # Accumulators
    AbstractAccumulator,
    accumulate_assume!!,
    accumulate_observe!!,
    accumulator_name,
    reset,
    split,
    combine,
    getacc,
    setacc!!,
    setaccs!!,
    deleteacc!!,
    VNTAccumulator,
    DoNotAccumulate,
    # Accumulators - logp
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    LogJacobianAccumulator,
    getlogjoint,
    getlogprior,
    getloglikelihood,
    getlogjac,
    getlogjoint_internal,
    getlogprior_internal,
    # Accumulators - values
    RawValueAccumulator,
    VectorValueAccumulator,
    VectorParamAccumulator,
    get_raw_values,
    get_vector_values,
    get_vector_params,
    # Accumulators - miscellany
    PriorDistributionAccumulator,
    BijectorAccumulator,
    # Working with internal values as vectors
    unflatten!!,
    internal_values_as_vector,
    # VarName (reexport from AbstractPPL)
    VarName,
    subsumes,
    @varname,
    # Compiler
    @model,
    # Utilities
    OrderedDict,
    typed_identity,
    # Model
    Model,
    getmissings,
    getargnames,
    setthreadsafe,
    requires_threadsafe,
    extract_priors,
    # evaluation
    evaluate!!,
    init!!,
    # LogDensityFunction
    LogDensityFunction,
    OnlyAccsVarInfo,
    to_vector_input,
    # Leaf contexts
    AbstractContext,
    contextualize,
    DefaultContext,
    InitContext,
    # Parent contexts
    AbstractParentContext,
    childcontext,
    setchildcontext,
    leafcontext,
    setleafcontext,
    # Tilde pipeline
    tilde_assume!!,
    tilde_observe!!,
    # Initialisation
    AbstractInitStrategy,
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    InitFromVector,
    get_param_eltype,
    init,
    # Transformed values
    VectorValue,
    LinkedVectorValue,
    UntransformedValue,
    get_transform,
    get_internal_value,
    set_internal_value,
    # Linking
    link,
    link!!,
    invlink,
    invlink!!,
    update_link_status!!,
    AbstractTransformStrategy,
    LinkAll,
    UnlinkAll,
    LinkSome,
    UnlinkSome,
    target_transform,
    apply_transform_strategy,
    AbstractTransform,
    DynamicLink,
    Unlink,
    # Pseudo distributions
    NamedDist,
    NoDist,
    # Convenience functions
    logprior,
    logjoint,
    pointwise_prior_logdensities,
    pointwise_logdensities,
    pointwise_loglikelihoods,
    condition,
    conditioned,
    decondition,
    fix,
    fixed,
    unfix,
    predict,
    marginalize,
    prefix,
    returned,
    to_submodel,
    # Struct to hold model outputs
    ParamsWithStats,
    # Convenience macros
    @addlogprob!,
    check_model,
    check_model_and_trace,
    # Deprecated.
    generated_quantities

# Reexport
using Distributions: loglikelihood
export loglikelihood

# TODO(mhauru) We should write down the list of methods that any subtype of AbstractVarInfo
# has to implement. Not sure what the full list is for parameters values, but for
# accumulators we only need `getaccs` and `setaccs!!`.
"""
    AbstractVarInfo

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

See also: [`VarInfo`](@ref)
"""
abstract type AbstractVarInfo <: AbstractModelTrace end

# Necessary forward declarations
include("utils.jl")
include("varnamedtuple.jl")
using .VarNamedTuples:
    VarNamedTuples,
    VarNamedTuple,
    map_pairs!!,
    map_values!!,
    apply!!,
    densify!!,
    templated_setindex!!,
    NoTemplate,
    SkipTemplate,
    @vnt

include("transformed_values.jl")
include("contexts.jl")
include("contexts/default.jl")
include("contexts/init.jl")
include("contexts/prefix.jl")
include("contexts/conditionfix.jl")  # Must come after contexts/prefix.jl
include("model.jl")
include("varname.jl")
include("distribution_wrappers.jl")
include("submodel.jl")
include("accumulators.jl")
include("accumulators/default.jl")
include("accumulators/vnt.jl")
include("accumulators/vector_values.jl")
include("accumulators/priors.jl")
include("accumulators/raw_values.jl")
include("accumulators/bijector.jl")
include("accumulators/pointwise_logdensities.jl")
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("onlyaccs.jl")
include("compiler.jl")
include("logdensityfunction.jl")
include("accumulators/vector_params.jl")
include("model_utils.jl")
include("chains.jl")

include("debug_utils.jl")
using .DebugUtils
include("test_utils.jl")

include("deprecated.jl")

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Same for MarginalLogDensities.jl
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            requires_mld =
                exc.f === DynamicPPL.marginalize &&
                length(argtypes) == 2 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractVector{<:Union{Symbol,<:VarName}}
            if requires_mld
                printstyled(
                    io,
                    "\n\n    `$(exc.f)` requires MarginalLogDensities.jl to be loaded.\n    Please run `using MarginalLogDensities` before calling `$(exc.f)`.\n";
                    color=:cyan,
                    bold=true,
                )
            end
        end
    end
end

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

# Extended in MarginalLogDensitiesExt
function marginalize end

end # module
