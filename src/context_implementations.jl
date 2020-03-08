alg_str(spl::Sampler) = string(nameof(typeof(spl.alg)))

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

_getindex(x, inds::Tuple) = _getindex(x[first(inds)...], Base.tail(inds))
_getindex(x, inds::Tuple{}) = x

# assume
function tilde(ctx::DefaultContext, sampler, right, vn::VarName, _, vi)
    return _tilde(sampler, right, vn, vi)
end
function tilde(ctx::PriorContext, sampler, right, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return _tilde(sampler, right, vn, vi)
end
function tilde(ctx::LikelihoodContext, sampler, right, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return _tilde(sampler, NoDist(right), vn, vi)
end
function tilde(ctx::MiniBatchContext, sampler, right, left::VarName, inds, vi)
    return tilde(ctx.ctx, sampler, right, left, inds, vi)
end

function _tilde(sampler, right, vn::VarName, vi)
    return Turing.assume(sampler, right, vn, vi)
end
function _tilde(sampler, right::NamedDist, vn::VarName, vi)
    name = right.name
    if name isa String
        sym_str, inds = split_var_str(name, String)
        sym = Symbol(sym_str)
        vn = VarName{sym}(inds)
    elseif name isa Symbol
        vn = VarName{name}("")
    elseif name isa VarName
        vn = name
    else
        throw("Unsupported variable name. Please use either a string, symbol or VarName.")
    end
    return _tilde(sampler, right.dist, vn, vi)
end

# observe
function tilde(ctx::DefaultContext, sampler, right, left, vi)
    return _tilde(sampler, right, left, vi)
end
function tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
function tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return _tilde(sampler, right, left, vi)
end
function tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * tilde(ctx.ctx, sampler, right, left, vi)
end

_tilde(sampler, right, left, vi) = Turing.observe(sampler, right, left, vi)

function assume(spl::Sampler, dist)
    error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    error("Turing.observe: unmanaged inference algorithm: $(typeof(spl))")
end

function assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo,
)
    if haskey(vi, vn)
        if is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = spl isa SampleFromUniform ? init(dist) : rand(dist)
            vi[vn] = vectorize(dist, r)
            setorder!(vi, vn, get_num_produce(vi))
        else
        r = vi[vn]
        end
    else
        r = isa(spl, SampleFromUniform) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, spl)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::Distribution,
    value,
    vi::VarInfo,
)
    increment_num_produce!(vi)
    return logpdf(dist, value)
end

# .~ functions

# assume
function dot_tilde(ctx::DefaultContext, sampler, right, left, vn::VarName, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    return _dot_tilde(sampler, dist, left, vns, vi)
end
function dot_tilde(
    ctx::LikelihoodContext,
    sampler,
    right,
    left,
    vn::VarName,
    inds,
    vi,
)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return _dot_tilde(sampler, NoDist(dist), left, vns, vi)
end
function dot_tilde(ctx::MiniBatchContext, sampler, right, left, vn::VarName, inds, vi)
    return dot_tilde(ctx.ctx, sampler, right, left, vn, inds, vi)
end
function dot_tilde(
    ctx::PriorContext,
    sampler,
    right,
    left,
    vn::VarName,
    inds,
    vi,
)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return _dot_tilde(sampler, dist, left, vns, vi)
end

function get_vns_and_dist(dist::NamedDist, var, vn::VarName)
    name = dist.name
    if name isa String
        sym_str, inds = split_var_str(name, String)
        sym = Symbol(sym_str)
        vn = VarName{sym}(inds)
    elseif name isa Symbol
        vn = VarName{name}("")
    elseif name isa VarName
        vn = name
    else
        throw("Unsupported variable name. Please use either a string, symbol or VarName.")
    end
    return get_vns_and_dist(dist.dist, var, vn)
end
function get_vns_and_dist(dist::MultivariateDistribution, var::AbstractMatrix, vn::VarName)
    getvn = i -> VarName(vn, vn.indexing * "[Colon(),$i]")
    return getvn.(1:size(var, 2)), dist
end
function get_vns_and_dist(
    dist::Union{Distribution, AbstractArray{<:Distribution}}, 
    var::AbstractArray, 
    vn::VarName
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    return getvn.(CartesianIndices(var)), dist
end

function _dot_tilde(sampler, right, left, vns::AbstractArray{<:VarName}, vi)
    return dot_assume(sampler, right, vns, left, vi)
end

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function _dot_tilde(
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution, AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vn::AbstractVector{<:VarName},
    vi::VarInfo,
)
    throw(ambiguity_error_msg())
end

function dot_assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::VarInfo,
)
    @assert length(dist) == size(var, 1)
    r = get_and_set_val!(vi, vns, dist, spl)
    lp = sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end
function dot_assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::VarInfo,
)
    r = get_and_set_val!(vi, vns, dists, spl)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end
function dot_assume(
    spl::Sampler,
    ::Any,
    ::AbstractArray{<:VarName},
    ::Any,
    ::VarInfo
)
    error("[Turing] $(alg_str(spl)) doesn't support vectorizing assume statement")
end

function get_and_set_val!(
    vi::VarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    spl::AbstractSampler,
)
    n = length(vns)
    if haskey(vi, vns[1])
        if is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
            r = spl isa SampleFromUniform ? init(dist, n) : rand(dist, n)
            for i in 1:n
                vn = vns[i]
                vi[vn] = vectorize(dist, r[:, i])
                setorder!(vi, vn, get_num_produce(vi))
            end
        else
        r = vi[vns]
        end
    else
        r = spl isa SampleFromUniform ? init(dist, n) : rand(dist, n)
        for i in 1:n
            push!(vi, vns[i], r[:,i], dist, spl)
        end
    end
    return r
end
function get_and_set_val!(
    vi::VarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    spl::AbstractSampler,
)
    if haskey(vi, vns[1])
        if is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
            f = (vn, dist) -> spl isa SampleFromUniform ? init(dist) : rand(dist)
            r = f.(vns, dists)
            for i in eachindex(vns)
                vn = vns[i]
                dist = dists isa AbstractArray ? dists[i] : dists
                vi[vn] = vectorize(dist, r[i])
                setorder!(vi, vn, get_num_produce(vi))
            end
        else
        r = reshape(vi[vec(vns)], size(vns))
        end
    else
        f = (vn, dist) -> spl isa SampleFromUniform ? init(dist) : rand(dist)
        r = f.(vns, dists)
        push!.(Ref(vi), vns, r, dists, Ref(spl))
    end
    return r
end

function set_val!(
    vi::VarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    val::AbstractMatrix,
)
    @assert size(val, 2) == length(vns)
    foreach(enumerate(vns)) do (i, vn)
        vi[vn] = val[:,i]
    end
    return val
end
function set_val!(
    vi::VarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    val::AbstractArray,
)
    @assert size(val) == size(vns)
    foreach(CartesianIndices(val)) do ind
        dist = dists isa AbstractArray ? dists[ind] : dists
        vi[vns[ind]] = vectorize(dist, val[ind])
    end
    return val
end

# observe
function dot_tilde(ctx::DefaultContext, sampler, right, left, vi)
    return _dot_tilde(sampler, right, left, vi)
end
function dot_tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
function dot_tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return _dot_tilde(sampler, right, left, vi)
end
function dot_tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * dot_tilde(ctx.ctx, sampler, right, left, left, vi)
end

function _dot_tilde(sampler, right, left::AbstractArray, vi)
    return dot_observe(sampler, right, left, vi)
end
# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function _dot_tilde(
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution, AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vi::VarInfo,
)
    throw(ambiguity_error_msg())
end

function dot_observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::MultivariateDistribution,
    value::AbstractMatrix,
    vi::VarInfo,
)
    increment_num_produce!(vi)
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "value = $value"
    return sum(logpdf(dist, value))
end
function dot_observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    increment_num_produce!(vi)
    Turing.DEBUG && @debug "dists = $dists"
    Turing.DEBUG && @debug "value = $value"
    return sum(logpdf.(dists, value))
end
function dot_observe(
    spl::Sampler,
    ::Any,
    ::Any,
    ::VarInfo,
)
    error("[Turing] $(alg_str(spl)) doesn't support vectorizing observe statement")
end
