using DynamicPPL:
    AbstractVarInfo,
    AccumulatorTuple,
    InitContext,
    InitFromVector,
    AbstractInitStrategy,
    LogJacobianAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    Model,
    ThreadSafeVarInfo,
    VarInfo,
    OnlyAccsVarInfo,
    RangeAndTransform,
    default_accumulators,
    float_type_with_fallback,
    getlogjoint,
    getlogjoint_internal,
    getloglikelihood,
    getlogprior,
    getlogprior_internal
using ADTypes: ADTypes
using BangBang: BangBang
using AbstractPPL: AbstractPPL, VarName
using LogDensityProblems: LogDensityProblems
using Random: Random

"""
    DynamicPPL.LogDensityFunction(
        model::Model,
        getlogdensity::Any,
        ranges_and_transforms::VarNamedTuple,
        x::AbstractVector{<:Real},
        accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
        fix_transforms::Bool=false,
    )

A struct which contains a model, along with all the information necessary to:

 - calculate its log density at a given point;
 - and if `adtype` is provided, calculate the gradient of the log density at that point.

This information can be extracted using the LogDensityProblems.jl interface, specifically,
using `LogDensityProblems.logdensity` and `LogDensityProblems.logdensity_and_gradient`. If
`adtype` is nothing, then only `logdensity` is implemented. If `adtype` is a concrete AD
backend type, then `logdensity_and_gradient` is also implemented.

## Positional arguments

!!! note
    You should almost never need to call this particular constructor of
    `LogDensityFunction`. Instead you should prefer the more convenient constructors that do
    not need an input vector `x`.

The first argument is the DynamicPPL model.

The second argument, `getlogdensity` should be a callable which takes a single argument: an
`OnlyAccsVarInfo`, and returns a `Real` corresponding to the log density of interest. There
are several functions in DynamicPPL that are 'supported' out of the box:

- [`getlogjoint_internal`](@ref): calculate the log joint, including the log-Jacobian term
  for any variables that have been linked in the provided VarInfo.
- [`getlogprior_internal`](@ref): calculate the log prior, including the log-Jacobian term
  for any variables that have been linked in the provided VarInfo.
- [`getlogjoint`](@ref): calculate the log joint in the model space, ignoring any effects of
  linking
- [`getlogprior`](@ref): calculate the log prior in the model space, ignoring any effects of
  linking
- [`getloglikelihood`](@ref): calculate the log likelihood (this is unaffected by linking,
  since transforms are only applied to random variables)

!!! note
    By default, `LogDensityFunction` uses `getlogjoint_internal`, i.e., the result of
    `LogDensityProblems.logdensity(f, x)` will depend on whether the `LogDensityFunction`
    was created with a linked or unlinked VarInfo. This is done primarily to ease
    interoperability with MCMC samplers.

The third argument is a VarNamedTuple which maps VarNames seen in the model to their
corresponding [`RangeAndTransform`](@ref). Each `RangeAndTransform`, as the name suggests,
contains a *range* which says which indices in the vectorised parameters correspond to that
variable, and a *transform* which says how to obtain the original (raw) value of that
variable from the slice.

The fourth argument is a sample vector of parameters, which should be consistent with the
ranges specified in the previous argument. This is used to determine the dimension and the
expected element type of the vectorised parameters, and is also used in AD preparation. The
values in the vector are not important.

!!! warning "Compiled ReverseDiff"
    For compiled ReverseDiff, the values in the vector are used to compile the tape, and so
    if your model has control flow that depends on the values of the parameters, then you
    may find that the resulting `LogDensityFunction` only yields correct results for parameters
    that trigger the same control flow as the sample vector. In general, functions with
    parameter-dependent control flow should not be differentiated with compiled ReverseDiff.

The last positional argument, `accs`, allows you to specify an `AccumulatorTuple` or a tuple
of `AbstractAccumulator`s which will be used _when evaluating the log density_`. (Note that
any accumulators from the previous argument are discarded.) This argument is not mandatory:
by default, this uses an internal function, `DynamicPPL.ldf_accs`, which attempts to choose
an appropriate set of accumulators based on which kind of log-density is being calculated.

## Keyword arguments

If the `adtype` keyword argument is provided, then this struct will also store the adtype
along with other information for efficient calculation of the gradient of the log density.
Note that preparing a `LogDensityFunction` with an AD type `AutoBackend()` requires the AD
backend itself to have been loaded (e.g. with `import Backend`).

## Fields

Note that it is undefined behaviour to access any of a `LogDensityFunction`'s fields, apart
from:

- `ldf.model`: The original model from which this `LogDensityFunction` was constructed.
- `ldf.adtype`: The AD type used for gradient calculations, or `nothing` if no AD
  type was provided.
- `ldf.transform_strategy`: The transform strategy that specifies the transforms for all
  variables in the model.

For all other fields, please use the corresponding getter functions provided in the API:

- [`get_logdensity_callable`](@ref)
- [`get_input_vector_type`](@ref)
- [`get_sample_input_vector`](@ref)
- [`get_range_and_transform`](@ref)
- [`get_all_ranges_and_transforms`](@ref)

# Extended help

Up until DynamicPPL v0.38, there have been two ways of evaluating a DynamicPPL model at a
given set of parameters:

1. With `unflatten!!` + `evaluate!!` with `DefaultContext`: this stores a vector of
   parameters inside a VarInfo's metadata, then reads parameter values from the VarInfo
   during evaluation.

2. With `InitFromParams`: this reads parameter values from a NamedTuple or a Dict, and
   stores them inside a VarInfo's metadata.

In general, both of these approaches work fine, but the fact that they modify the VarInfo's
metadata can often be quite wasteful. In particular, it is very common that the only outputs
we care about from model evaluation are those which are stored in accumulators, such as log
probability densities, or raw values.

To avoid this issue, we use `OnlyAccsVarInfo`, which is a VarInfo that only contains
accumulators. It implements enough of the `AbstractVarInfo` interface to not error during
model evaluation.

Because `OnlyAccsVarInfo` does not store any parameter values, when evaluating a model with
it, it is mandatory that parameters are provided from outside the VarInfo, namely via
`InitContext`.

The main problem that we face is that it is not possible to directly implement
`DynamicPPL.init(rng, vn, dist, strategy)` for `strategy::InitFromParams{<:AbstractVector}`.
In particular, it is not clear:

 - which parts of the vector correspond to which random variables, and
 - whether the variables are linked or unlinked.

Traditionally, this problem has been solved by `unflatten!!`, because that function would
place values into the VarInfo's metadata alongside the information about ranges and linking.
That way, when we evaluate with `DefaultContext`, we can read this information out again.
However, we want to avoid using a metadata. Thus, here, we _extract this information from
the VarInfo_ a single time when constructing a `LogDensityFunction` object. Inside the
LogDensityFunction, we store a mapping from VarNames to ranges in that vector, along with
link status.

When evaluating the model, this allows us to combine the parameter vector together with
those ranges to create an `InitFromVector`, which lets us very quickly read parameter values
from the vector.

Note that this assumes that the ranges and link status are static throughout the lifetime of
the `LogDensityFunction` object. Therefore, a `LogDensityFunction` object cannot handle
models which have variable numbers of parameters, or models which may visit random variables
in different orders depending on stochastic control flow. **Indeed, silent errors may occur
with such models.** This is a general limitation of vectorised parameters: the original
`unflatten!!` + `evaluate!!` approach also fails with such models.
"""
struct LogDensityFunction{
    M<:Model,
    AD<:Union{ADTypes.AbstractADType,Nothing},
    L<:AbstractTransformStrategy,
    F,
    VNT<:VarNamedTuple,
    ADP,
    # type of the vector passed to logdensity functions
    X<:AbstractVector,
    AC<:AccumulatorTuple,
    # whether all transforms are FixedTransforms
    AllFixed,
}
    model::M
    adtype::AD
    transform_strategy::L
    _getlogdensity::F
    _varname_ranges::VNT
    _adprep::ADP
    _dim::Int
    _x::X
    _accs::AC

    function LogDensityFunction(
        model::Model,
        getlogdensity::Any,
        ranges_and_transforms::VarNamedTuple,
        x::AbstractVector{<:Real},
        accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(
            getlogdensity
        );
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    )
        dim = length(x)
        # Determine LDF transform strategy.
        transform_strategy = infer_transform_strategy_from_values(ranges_and_transforms)
        # Determine whether all transforms are fixed. This enables fast parameter
        # extraction in ParamsWithStats without model re-evaluation.
        all_fixed = all(
            rat -> rat.transform isa FixedTransform, values(ranges_and_transforms)
        )
        # convert to AccumulatorTuple if needed
        accs = AccumulatorTuple(accs)
        # Do AD prep if needed
        prep = if adtype === nothing
            nothing
        else
            # Make backend-specific tweaks to the adtype
            adtype = DynamicPPL.tweak_adtype(adtype, model, x)
            problem = LogDensityAt(
                model, getlogdensity, ranges_and_transforms, transform_strategy, accs
            )
            # `x` was just constructed from the same range metadata stored in `problem`,
            # so the AD wrapper can skip its hot-path dimension validation.
            AbstractPPL.prepare(adtype, problem, x; check_dims=false)
        end
        return new{
            typeof(model),
            typeof(adtype),
            typeof(transform_strategy),
            typeof(getlogdensity),
            typeof(ranges_and_transforms),
            typeof(prep),
            typeof(x),
            typeof(accs),
            all_fixed,
        }(
            model,
            adtype,
            transform_strategy,
            getlogdensity,
            ranges_and_transforms,
            prep,
            dim,
            x,
            accs,
        )
    end
end

"""
    DynamicPPL.LogDensityFunction(
        model::Model,
        getlogdensity::Any=getlogjoint_internal,
        vecvals_or_strategy=UnlinkAll(),
        accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
        adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
        fix_transforms::Bool=false,
    )

Most users of LogDensityFunction should use this constructor, which does **not** require
passing a sample input vector `x`.

The first two arguments are the same as in the four-argument constructor.

- `model` is the DynamicPPL model for which we want to construct a LogDensityFunction.
- `getlogdensity` is a callable which takes a single argument: an `OnlyAccsVarInfo`, and
  returns a `Real` corresponding to the log density of interest. Most of the time this is
  `getlogjoint_internal`.

**The third argument** can take many forms, but it essentially encodes all the necessary
information to generate the `RangeAndTransform`s, as well as the sample input vector `x`
that is required for the four-argument constructor.

You can pass either:

- **`vnt`**: a `VarNamedTuple` which contains vectorised representations of all the random
  variables in the model (i.e., it maps `VarName`s to
  `TransformedValue{<:AbstractVector}`s). This is useful if you already have one, either by
  creating a full `VarInfo` and accessing its `values` field, or by creating a
  `OnlyAccsVarInfo` with a `VectorValueAccumulator` and calling `get_vector_values` on it.

- **`vi`**: a `VarInfo`, in which case the `vi.values` field is used.

- **`oavi`**: an [`OnlyAccsVarInfo`](@ref), in which case the [`get_vector_values`](@ref)
  function is used to extract a VarNamedTuple of vector values from the
  [`VectorValueAccumulator`](@ref) inside it. If the `OnlyAccsVarInfo` does not contain a
  `VectorValueAccumulator`, then an error is thrown.

- **`transform_strategy`**: *by far the most convenient way*. In this case, the transform
  strategy will be first used to generate a set of vectorised parameters, from which the
  relevant information will be extracted. This does come at the cost of doing one extra
  model evaluation. Whilst `LogDensityFunction` construction is unlikely to occur in
  performance-sensitive code paths, if you absolutely cannot pay this price, then you should
  generate the vectorised parameters yourself and pass them here instead.

## Keyword arguments

The `adtype` keyword argument allows you to specify an AD type for gradient preparation and
calculation.

The `fix_transforms` keyword argument allows you to specify whether the transforms used in
the `LogDensityFunction` should be cached at the time of construction. If so, the model is
evaluated once using the provided transform strategy, and the transforms used for each
variable are stored in the `LogDensityFunction`. This allows you to avoid the overhead of
recalculating transforms during each log-density evaluation. See [the documentation on fixed
transforms](@ref fixed-transforms) for more information.
"""
function LogDensityFunction(
    model::Model,
    getlogdensity::Any,
    # This VNT should map varnames to TransformedValue{<:AbstractVector}
    vecvals::VarNamedTuple,
    accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
    adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    fix_transforms::Bool=false,
)
    # Handle fixed transforms flag.
    if fix_transforms
        all_fixed = all(tv -> get_transform(tv) isa FixedTransform, values(vecvals))
        if !all_fixed
            # We need to update the transforms in `vnt` to be consistent with the new
            # transform strategy. This requires reevaluating the model in
            # `get_fixed_transforms`, which is perhaps a bit unfortunate, but probably
            # tolerable since this isn't something that is in a performance-sensitive code
            # path.
            dynamic_transform_strategy = infer_transform_strategy_from_values(vecvals)
            transforms_vnt = get_fixed_transforms(model, dynamic_transform_strategy)
            vecvals = update_transforms!!(vecvals, transforms_vnt)
        end
    end
    ranges_and_transforms, x = get_rat_and_samplevec(vecvals)
    return LogDensityFunction(
        model, getlogdensity, ranges_and_transforms, x, accs; adtype=adtype
    )
end
function LogDensityFunction(
    model::Model,
    getlogdensity::Any,
    vi::VarInfo,
    accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
    adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    fix_transforms::Bool=false,
)
    return LogDensityFunction(
        model, getlogdensity, vi.values, accs; adtype=adtype, fix_transforms=fix_transforms
    )
end
function LogDensityFunction(
    model::Model,
    getlogdensity::Any,
    oavi::OnlyAccsVarInfo,
    accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
    adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    fix_transforms::Bool=false,
)
    if !hasacc(oavi, Val(VECTORVAL_ACCNAME))
        error(
            "When constructing a LogDensityFunction with an OnlyAccsVarInfo, you must include a VectorValueAccumulator as one of the accumulators, so that the parameter vector can be extracted from the VarInfo. The provided OnlyAccsVarInfo does not have a VectorValueAccumulator.",
        )
    end
    vnt = getacc(oavi, Val(VECTORVAL_ACCNAME)).values
    return LogDensityFunction(
        model, getlogdensity, vnt, accs; adtype=adtype, fix_transforms=fix_transforms
    )
end
function LogDensityFunction(
    model::Model,
    getlogdensity::Any=getlogjoint_internal,
    transform_strategy::AbstractTransformStrategy=UnlinkAll(),
    accs::Union{NTuple{<:Any,AbstractAccumulator},AccumulatorTuple}=ldf_accs(getlogdensity);
    adtype::Union{ADTypes.AbstractADType,Nothing}=nothing,
    fix_transforms::Bool=false,
)
    # note that this reevaluates the model
    oavi = OnlyAccsVarInfo(VectorValueAccumulator())
    _, oavi = DynamicPPL.init!!(model, oavi, InitFromPrior(), transform_strategy)
    vecvals = getacc(oavi, Val(VECTORVAL_ACCNAME)).values
    return LogDensityFunction(
        model, getlogdensity, vecvals, accs; adtype=adtype, fix_transforms=fix_transforms
    )
end

"""
    DynamicPPL.get_input_vector_type(::LogDensityFunction)

Get the type of the vector `x` that should be passed to `LogDensityProblems.logdensity(ldf,
x)`.

Note that if you pass a vector of a different type, it will be converted to the correct
type. This allows you however to determine upfront what kind of vector should be passed in.
It is also useful for determining e.g. whether Float32 or Float64 parameters are expected.
"""
get_input_vector_type(::LogDensityFunction{M,A,L,G,R,P,X}) where {M,A,L,G,R,P,X} = X

"""
    DynamicPPL.get_sample_input_vector(::LogDensityFunction)::AbstractVector{<:Real}

Get the sample input vector `x` used to construct the LogDensityFunction.
"""
get_sample_input_vector(ldf::LogDensityFunction) = ldf._x

"""
    DynamicPPL.get_range_and_transform(ldf::LogDensityFunction, vn::VarName)::RangeAndTransform

A `LogDensityFunction` stores a mapping from `VarName`s to their corresponding ranges in the
vectorised parameter representation, along with their transform status. This function
retrieves that information for a single VarName.
"""
get_range_and_transform(ldf::LogDensityFunction, vn::VarName) = ldf._varname_ranges[vn]

"""
    DynamicPPL.get_all_ranges_and_transforms(ldf::LogDensityFunction)::VarNamedTuple

A `LogDensityFunction` stores a mapping from `VarName`s to their corresponding ranges in the
vectorised parameter representation, along with their transform status. This function
retrieves the complete mapping.
"""
get_all_ranges_and_transforms(ldf::LogDensityFunction) = ldf._varname_ranges

"""
    DynamicPPL.get_logdensity_callable(ldf::LogDensityFunction)

A `LogDensityFunction` stores a callable that, given an `OnlyAccsVarInfo`, can be used to
calculate the log density of the model at a given set of parameters. For example, most
usecases in DynamicPPL use [`DynamicPPL.getlogjoint_internal`](@ref) for this purpose.

This function retrieves that callable.
"""
get_logdensity_callable(l::LogDensityFunction) = l._getlogdensity

###################################
# LogDensityProblems.jl interface #
###################################
"""
    ldf_accs(getlogdensity::Any)

Determine which accumulators are needed for fast evaluation with the given
`getlogdensity` callable.
"""
ldf_accs(::Any) = default_accumulators()
ldf_accs(::typeof(getlogjoint_internal)) = default_accumulators()
function ldf_accs(::typeof(getlogjoint))
    return AccumulatorTuple((LogPriorAccumulator(), LogLikelihoodAccumulator()))
end
function ldf_accs(::typeof(getlogprior_internal))
    return AccumulatorTuple((LogPriorAccumulator(), LogJacobianAccumulator()))
end
ldf_accs(::typeof(getlogprior)) = AccumulatorTuple((LogPriorAccumulator(),))
ldf_accs(::typeof(getloglikelihood)) = AccumulatorTuple((LogLikelihoodAccumulator(),))

"""
    logdensity_at(
        params::AbstractVector{<:Real},
        model::Model,
        getlogdensity::Any,
        varname_ranges::VarNamedTuple,
        transform_strategy::AbstractTransformStrategy,
        accs::AccumulatorTuple,
    )

Calculate the log density at the given `params`, using the provided information extracted
from a `LogDensityFunction`.
"""
function logdensity_at(
    params::AbstractVector{<:Real},
    model::Model,
    getlogdensity::Any,
    varname_ranges::VarNamedTuple,
    transform_strategy::AbstractTransformStrategy,
    accs::AccumulatorTuple,
)
    init_strategy = InitFromVector(params, varname_ranges, transform_strategy)
    _, vi = DynamicPPL.init!!(
        model, OnlyAccsVarInfo(accs), init_strategy, transform_strategy
    )
    return getlogdensity(vi)
end

"""
    LogDensityAt(
        model::Model,
        getlogdensity::Any,
        varname_ranges::VarNamedTuple,
        transform_strategy::AbstractTransformStrategy,
        accs::AccumulatorTuple,
    )

A callable struct that behaves in the same way as `logdensity_at`, but stores the model and
other information internally. Having two separate functions/structs allows for better
performance with AD backends.
"""
struct LogDensityAt{
    M<:Model,F,V<:VarNamedTuple,L<:AbstractTransformStrategy,A<:AccumulatorTuple
}
    model::M
    getlogdensity::F
    varname_ranges::V
    transform_strategy::L
    accs::A

    function LogDensityAt(
        model::M, getlogdensity::F, varname_ranges::V, transform_strategy::L, accs::A
    ) where {M,F,V,L,A}
        return new{M,F,V,L,A}(
            model, getlogdensity, varname_ranges, transform_strategy, accs
        )
    end
end
function (f::LogDensityAt)(params::AbstractVector{<:Real})
    return logdensity_at(
        params, f.model, f.getlogdensity, f.varname_ranges, f.transform_strategy, f.accs
    )
end

@inline function LogDensityProblems.logdensity(
    ldf::LogDensityFunction, params::AbstractVector{<:Real}
)
    return logdensity_at(
        params,
        ldf.model,
        ldf._getlogdensity,
        ldf._varname_ranges,
        ldf.transform_strategy,
        ldf._accs,
    )
end

@inline function LogDensityProblems.logdensity_and_gradient(
    ldf::LogDensityFunction, params::AbstractVector{<:Real}
)
    # `params` has to be converted to the same vector type that was used for AD preparation,
    # otherwise the preparation will not be valid.
    params = convert(get_input_vector_type(ldf), params)
    return AbstractPPL.value_and_gradient!!(ldf._adprep, params)
end

function LogDensityProblems.capabilities(::Type{<:LogDensityFunction{M,Nothing}}) where {M}
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(
    ::Type{<:LogDensityFunction{M,<:ADTypes.AbstractADType}}
) where {M}
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.dimension(ldf::LogDensityFunction)
    return ldf._dim
end

"""
    tweak_adtype(
        adtype::ADTypes.AbstractADType,
        model::Model,
        params::AbstractVector
    )

Return an 'optimised' form of the adtype. This is useful for doing backend-specific
optimisation of the adtype (e.g., for ForwardDiff, calculating the chunk size: see the
method override in `ext/DynamicPPLForwardDiffExt.jl`). The model is passed as a parameter in
case the optimisation depends on the model.

By default, this just returns the input unchanged.
"""
tweak_adtype(adtype::ADTypes.AbstractADType, ::Model, ::AbstractVector) = adtype

######################################################
# Helper functions to extract ranges and link status #
######################################################

"""
    get_rat_and_samplevec(vnt::VarNamedTuple)

Given a `VarNamedTuple` that contains vectorised values (i.e.,
`TransformedValue{<:AbstractVector}`), extract the ranges of each variable in the vectorised
parameter representation. Further infer the transform status of each variable from the type
of the vectorised value.

This function returns a VarNamedTuple mapping all VarNames to their corresponding
`RangeAndTransform`, plus a vector of parameters obtained by concatenating all the
vectorised values together.
"""
function get_rat_and_samplevec(vnt::VarNamedTuple)
    # Note: can't use map_values!! here as that might mutate the VNT itself!
    ranges_vnt, x, _ = mapreduce(
        identity,
        function ((ranges_vnt, params, offset), pair)
            vn, tv = pair
            val = get_internal_value(tv)
            if !(val isa AbstractVector)
                throw(
                    ArgumentError(
                        "Expected all values in the provided VarNamedTuple to be TransformedValues wrapping AbstractVectors, but the value for variable `$vn` is a $(typeof(val)).",
                    ),
                )
            end
            range = offset:(offset + length(val) - 1)
            offset += length(val)
            params = vcat(params, val)
            ral = RangeAndTransform(range, tv.transform)
            template = vnt.data[AbstractPPL.getsym(vn)]
            ranges_vnt = templated_setindex!!(ranges_vnt, ral, vn, template)
            return ranges_vnt, params, offset
        end,
        vnt;
        init=(VarNamedTuple(), Union{}[], 1),
    )
    return ranges_vnt, x
end

"""
    update_transforms!!(vnt::VarNamedTuple, transforms_vnt::VarNamedTuple)

Given `vnt` which contains vectorised values (i.e., `TransformedValue{<:AbstractVector}`), and a
`transforms_vnt` which contains the transforms to be applied to each variable, update the
vectorised values in `vnt` to have the corresponding transforms from `transforms_vnt`.

This function returns a VarNamedTuple mapping all VarNames to their corresponding
`TransformedValue`s.

!!! warning
    This function might mutate `vnt`.
"""
function update_transforms!!(vnt::VarNamedTuple, transforms_vnt::VarNamedTuple)
    return map_pairs!!(vnt) do pair
        vn, tv = pair
        TransformedValue(get_internal_value(tv), transforms_vnt[vn])
    end
end

"""
    InitFromVector(
        vect::AbstractVector{<:Real},
        ldf::LogDensityFunction
    )

Constructor for `InitFromVector` that extracts the necessary information about VarName
ranges and transform strategy from a pre-existing `LogDensityFunction`.
"""
function InitFromVector(
    vect::V, ldf::L
) where {V<:AbstractVector{<:Real},L<:LogDensityFunction}
    varname_ranges = ldf._varname_ranges
    transform_strategy = ldf.transform_strategy
    if length(vect) != ldf._dim
        throw(
            ArgumentError(
                "The length of the input vector is $(length(vect)), but the LogDensityFunction expects a vector of length $(ldf._dim) based on the ranges that were extracted when the LogDensityFunction was constructed.",
            ),
        )
    end
    return InitFromVector(vect, varname_ranges, transform_strategy)
end

"""
    to_vector_params(
        vector_values::VarNamedTuple,
        ldf::LogDensityFunction
    )

Extract vectorised values from a `VarNamedTuple`, and concatenate them into a single vector
that is consistent with the ranges specified in the `LogDensityFunction`.

This is useful when you want to regenerate new vectorised parameters but using a different
initialisation strategy.

Note that the transform status of the variables in the `VarNamedTuple` must be consistent
with the transform strategy stored in the `LogDensityFunction`. This function checks for
that.
"""
function to_vector_params(vector_values::VarNamedTuple, ldf::LogDensityFunction)
    return to_vector_params_inner(
        vector_values, ldf._varname_ranges, eltype(get_input_vector_type(ldf)), ldf._dim
    )
end

function to_vector_params_inner(
    vector_values::VarNamedTuple, ranges::VarNamedTuple, ::Type{eltype}, dim::Int
) where {eltype}
    template_vect = Vector{eltype}(undef, dim)

    # We want to make sure that every element in `template_vect` is written to exactly once.
    set_indices = similar(template_vect, Bool)
    fill!(set_indices, false)
    for (vn, tval) in pairs(vector_values)
        if !haskey(ranges, vn)
            throw(
                ArgumentError(
                    "The variable `$vn` is present in the provided VarNamedTuple of vector values, but there is no record of this in the LogDensityFunction. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
                ),
            )
        end
        ral = ranges[vn]

        # check transform lines up
        if ral.transform != tval.transform
            throw(
                ArgumentError(
                    "The variable `$vn` has transform status $(ral.transform) in the LogDensityFunction, but the provided VarNamedTuple has transform status $(tval.transform) for this variable. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
                ),
            )
        end

        # Get the internal vector and check its length
        vec_val = DynamicPPL.get_internal_value(tval)
        len = length(vec_val)
        expected_len = length(ral.range)
        if len != expected_len
            throw(
                ArgumentError(
                    "The length of the vector value provided for `$vn` is $len, but the LogDensityFunction expects it to be $expected_len based on the ranges that were extracted when the LogDensityFunction was constructed.",
                ),
            )
        end

        # Set it
        # TODO(penelopeysm): can we use views? Does it make a difference?
        if any(set_indices[ral.range])
            throw(
                ArgumentError(
                    "Setting to the same indices in the output vector more than once. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
                ),
            )
        end
        BangBang.setindex!!(template_vect, vec_val, ral.range)
        set_indices[ral.range] .= true
    end

    # Once we're done, we should check that all values were set
    if !all(set_indices)
        throw(
            ArgumentError(
                "Some indices in the output vector were not set. This likely means that the vector values provided are not consistent with the LogDensityFunction (e.g. if they were obtained from a different model).",
            ),
        )
    end

    return template_vect
end

"""
    Base.rand(
        [rng::AbstractRNG,]
        ldf::LogDensityFunction,
        init_strategy::AbstractInitStrategy=InitFromPrior(),
    )

Generate a random vector of parameters that is consistent with the given
`LogDensityFunction`, using the provided initialisation strategy.

Note that this function only generates parameters, and does not return the log density. If
you also need the log density, instead of calling `rand` and then
`LogDensityProblems.logdensity` separately (which will incur two separate model
evaluations), you can directly call `DynamicPPL.init!!` with a
[`VectorParamAccumulator`](@ref) plus the log-probability accumulators that you need, and
extract both the parameters and the log density from the resulting `OnlyAccsVarInfo`. See
[the DynamicPPL documentation](@ref ldf-model) for an example of this.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    accs = OnlyAccsVarInfo(VectorParamAccumulator(ldf))
    _, accs = DynamicPPL.init!!(rng, ldf.model, accs, init_strategy, ldf.transform_strategy)
    return get_vector_params(accs)
end
function Base.rand(
    ldf::LogDensityFunction, init_strategy::AbstractInitStrategy=InitFromPrior()
)
    return rand(Random.default_rng(), ldf, init_strategy)
end
