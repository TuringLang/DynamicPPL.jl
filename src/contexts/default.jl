"""
    DefaultContext(values::VarNamedTuple=VarNamedTuple())

Evaluate using the vectorised `TransformedValue`s supplied in `values`.

Each value carries its transform information. The output varinfo does not supply latent
values or determine their transforms. Use [`InitContext`](@ref) to obtain new values or
choose a different transform strategy.

The context borrows `values`; it does not copy the underlying parameter arrays.

# Examples

```jldoctest
julia> @model example() = x ~ Normal();

julia> previous = VarInfo(example(), InitFromParams((; x=1.0)));

julia> context = DefaultContext(get_values(previous));

julia> retval, outputs = evaluate!!(example(), context, OnlyAccsVarInfo());

julia> retval
1.0
```
"""
struct DefaultContext{V<:VarNamedTuple} <: AbstractContext
    values::V
    function DefaultContext(values::V=VarNamedTuple()) where {V<:VarNamedTuple}
        mapreduce(
            pair ->
                pair.second isa TransformedValue{
                    <:AbstractVector{<:Real},<:Union{DynamicLink,Unlink,FixedTransform}
                },
            &,
            values;
            init=true,
        ) || throw(ArgumentError("DefaultContext requires vectorised TransformedValues"))
        return new{V}(values)
    end
end

get_transformed_value(context::DefaultContext, vn::VarName) = context.values[vn]
function get_param_eltype(context::DefaultContext)
    return mapreduce(
        pair -> eltype(get_internal_value(pair.second)),
        promote_type,
        context.values;
        init=Union{},
    )
end

"""
    DynamicPPL.tilde_assume!!(
        context::DefaultContext,
        right::Distribution,
        vn::VarName,
        template::Any,
        vi::AbstractVarInfo
    )

Read `vn` from the context, store it in the output varinfo, and accumulate its log probability.
Throw `KeyError` if the context does not supply the variable.
"""
function tilde_assume!!(
    context::DefaultContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::AbstractVarInfo,
)
    tval = get_transformed_value(context, vn)
    trf = if tval.transform isa DynamicLink
        Bijectors.VectorBijectors.from_linked_vec(right)
    elseif tval.transform isa Unlink
        Bijectors.VectorBijectors.from_vec(right)
    elseif tval.transform isa FixedTransform
        tval.transform.transform
    else
        error("Expected transformed value to be a vectorised value")
    end
    x, inv_logjac = with_logabsdet_jacobian(trf, get_internal_value(tval))
    vi = setindex_with_dist!!(vi, tval, right, vn, template)
    vi = accumulate_assume!!(vi, x, tval, -inv_logjac, vn, right, template)
    return x, vi
end
