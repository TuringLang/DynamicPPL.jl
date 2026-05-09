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
    get_values,
    VarNamedTuple,
    @vnt,
    map_pairs!!,
    map_values!!,
    apply!!,
    densify!!,
    skeleton,
    push!!,
    empty!!,
    SkipTemplate,
    NoTemplate,
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
    get_priors,
    FixedTransformAccumulator,
    get_fixed_transforms,
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
    # Model
    Model,
    setthreadsafe,
    requires_threadsafe,
    extract_priors,
    # evaluation
    evaluate!!,
    init!!,
    # LogDensityFunction
    LogDensityFunction,
    OnlyAccsVarInfo,
    to_vector_params,
    get_input_vector_type,
    get_sample_input_vector,
    RangeAndTransform,
    get_range_and_transform,
    get_all_ranges_and_transforms,
    get_logdensity_callable,
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
    extract_prefixes,
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
    TransformedValue,
    get_transform,
    get_internal_value,
    get_raw_value,
    set_internal_value,
    # Transform strategies
    update_transform_status!!,
    AbstractTransformStrategy,
    LinkAll,
    UnlinkAll,
    LinkSome,
    UnlinkSome,
    WithTransforms,
    target_transform,
    apply_transform_strategy,
    AbstractTransform,
    DynamicLink,
    Unlink,
    FixedTransform,
    NoTransform,
    # Linking
    link,
    link!!,
    invlink,
    invlink!!,
    # Pseudo distributions
    NamedDist,
    NoDist,
    filldist,
    arraydist,
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
    prefix,
    returned,
    to_submodel,
    # Struct to hold model outputs
    ParamsWithStats,
    # Convenience macros
    @addlogprob!,
    check_model,
    set_logprob_type!,
    # Deprecated.
    generated_quantities,
    typed_identity

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

See also: [`VarInfo`](@ref), [`OnlyAccsVarInfo`](@ref).
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
    @vnt,
    skeleton

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
include("accumulators/fixed_transforms.jl")
include("accumulators/pointwise_logdensities.jl")
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("onlyaccs.jl")
include("compiler.jl")
include("logdensityfunction.jl")
include("accumulators/vector_params.jl")
include("chains.jl")

include("debug_utils.jl")
using .DebugUtils
include("test_utils.jl")

include("deprecated.jl")

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

end # module
