"""
Context that records logp after tilde_assume!!
for each VarName used by [`varwise_logpriors`](@ref).
"""
struct VarwisePriorContext{A,Ctx} <: AbstractContext
    logpriors::A
    context::Ctx
end

function VarwisePriorContext(
    logpriors=OrderedDict{Symbol,Float64}(),
    context::AbstractContext=DynamicPPL.PriorContext(),
    #OrderedDict{Symbol,Vector{Float64}}(),PriorContext()),
)
    return VarwisePriorContext{typeof(logpriors),typeof(context)}(
        logpriors, context
    )
end

NodeTrait(::VarwisePriorContext) = IsParent()
childcontext(context::VarwisePriorContext) = context.context
function setchildcontext(context::VarwisePriorContext, child)
    return VarwisePriorContext(context.logpriors, child)
end

function tilde_assume(context::VarwisePriorContext, right, vn, vi)
    #@info "VarwisePriorContext tilde_assume!! called for $vn"
    value, logp, vi = tilde_assume(context.context, right, vn, vi)
    #sym = DynamicPPL.getsym(vn)
    new_context = acc_logp!(context, vn, logp)
    return value, logp, vi
end

function dot_tilde_assume(context::VarwisePriorContext, right, left, vn, vi)
    #@info "VarwisePriorContext dot_tilde_assume!! called for $vn"
    # @show vn, left, right, typeof(context).name
    value, logp, vi = dot_tilde_assume(context.context, right, left, vn, vi)
    new_context = acc_logp!(context, vn, logp)
    return value, logp, vi
end


function tilde_observe(context::VarwisePriorContext, right, left, vi)
    # Since we are evaluating the prior, the log probability of all the observations
    # is set to 0. This has the effect of ignoring the likelihood.
    return 0.0, vi
    #tmp = tilde_observe(context.context, SampleFromPrior(), right, left, vi)
    #return tmp
end

function acc_logp!(context::VarwisePriorContext, vn::Union{VarName,AbstractVector{<:VarName}}, logp)
    #sym = DynamicPPL.getsym(vn)  # leads to duplicates
    # if vn is a Vector leads to Symbol("VarName{:s, IndexLens{Tuple{Int64}}}[s[1], s[2]]")    
    sym = Symbol(vn) 
    context.logpriors[sym] = logp
    return (context)
end


# """
#     pointwise_logpriors(model::Model, chain::Chains, keytype = String)

# Runs `model` on each sample in `chain` returning a `OrderedDict{String, Matrix{Float64}}`
# with keys corresponding to symbols of the observations, and values being matrices
# of shape `(num_chains, num_samples)`.

# `keytype` specifies what the type of the keys used in the returned `OrderedDict` are.
# Currently, only `String` and `VarName` are supported.

# # Notes
# Say `y` is a `Vector` of `n` i.i.d. `Normal(μ, σ)` variables, with `μ` and `σ`
# both being `<:Real`. Then the *observe* (i.e. when the left-hand side is an
# *observation*) statements can be implemented in three ways:
# 1. using a `for` loop:
# ```julia
# for i in eachindex(y)
#     y[i] ~ Normal(μ, σ)
# end
# ```
# 2. using `.~`:
# ```julia
# y .~ Normal(μ, σ)
# ```
# 3. using `MvNormal`:
# ```julia
# y ~ MvNormal(fill(μ, n), σ^2 * I)
# ```

# In (1) and (2), `y` will be treated as a collection of `n` i.i.d. 1-dimensional variables,
# while in (3) `y` will be treated as a _single_ n-dimensional observation.

# This is important to keep in mind, in particular if the computation is used
# for downstream computations.

# # Examples
# ## From chain
# ```julia-repl
# julia> using DynamicPPL, Turing

# julia> @model function demo(xs, y)
#            s ~ InverseGamma(2, 3)
#            m ~ Normal(0, √s)
#            for i in eachindex(xs)
#                xs[i] ~ Normal(m, √s)
#            end

#            y ~ Normal(m, √s)
#        end
# demo (generic function with 1 method)

# julia> model = demo(randn(3), randn());

# julia> chain = sample(model, MH(), 10);

# julia> pointwise_logpriors(model, chain)
# OrderedDict{String,Array{Float64,2}} with 4 entries:
#   "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
#   "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
#   "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
#   "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

# julia> pointwise_logpriors(model, chain, String)
# OrderedDict{String,Array{Float64,2}} with 4 entries:
#   "xs[1]" => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
#   "xs[2]" => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
#   "xs[3]" => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
#   "y"     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]

# julia> pointwise_logpriors(model, chain, VarName)
# OrderedDict{VarName,Array{Float64,2}} with 4 entries:
#   xs[1] => [-1.42932; -2.68123; … ; -1.66333; -1.66333]
#   xs[2] => [-1.6724; -0.861339; … ; -1.62359; -1.62359]
#   xs[3] => [-1.42862; -2.67573; … ; -1.66251; -1.66251]
#   y     => [-1.51265; -0.914129; … ; -1.5499; -1.5499]
# ```

# ## Broadcasting
# Note that `x .~ Dist()` will treat `x` as a collection of
# _independent_ observations rather than as a single observation.

# ```jldoctest; setup = :(using Distributions)
# julia> @model function demo(x)
#            x .~ Normal()
#        end;

# julia> m = demo([1.0, ]);

# julia> ℓ = pointwise_logpriors(m, VarInfo(m)); first(ℓ[@varname(x[1])])
# -1.4189385332046727

# julia> m = demo([1.0; 1.0]);

# julia> ℓ = pointwise_logpriors(m, VarInfo(m)); first.((ℓ[@varname(x[1])], ℓ[@varname(x[2])]))
# (-1.4189385332046727, -1.4189385332046727)
# ```

# """
function varwise_logpriors(
    model::Model, varinfo::AbstractVarInfo,
    context::AbstractContext=PriorContext()
)
#    top_context = VarwisePriorContext(OrderedDict{Symbol,Float64}(), context)
    top_context = VarwisePriorContext(OrderedDict{Symbol,Float64}(), context)
    model(varinfo, top_context)
    return top_context.logpriors
end

function varwise_logpriors(model::Model, chain::AbstractChains,
    context::AbstractContext=PriorContext();
    top_context::VarwisePriorContext{T} = VarwisePriorContext(OrderedDict{Symbol,Float64}(), context)
    ) where T
    # pass top-context as keyword to allow adapt Number type of log-prior
    get_values = (vi) -> begin
        model(vi, top_context)
        values(top_context.logpriors)
    end
    arr =  map_model(get_values, model, chain)
    par_names = collect(keys(top_context.logpriors))
    return(arr, par_names)
end

function map_model(get_values, model::Model, chain::AbstractChains)
    niters = size(chain, 1)
    nchains = size(chain, 3)
    vi = VarInfo(model)
    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    # initialize the array by the first result
    (sample_idx, chain_idx), iters2 = Iterators.peel(iters)
    setval!(vi, chain, sample_idx, chain_idx)
    values1 = get_values(vi)
    arr = Array{eltype(values1)}(undef, niters, length(values1), nchains)
    arr[sample_idx, :, chain_idx] .= values1
    #(sample_idx, chain_idx), iters3 = Iterators.peel(iters2)
    for (sample_idx, chain_idx) in iters2
        # Update the values
        setval!(vi, chain, sample_idx, chain_idx)
        values_i = get_values(vi)
        arr[sample_idx, :, chain_idx] .= values_i
    end
    return(arr)
end
