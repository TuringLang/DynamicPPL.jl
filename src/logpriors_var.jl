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


tilde_observe(context::VarwisePriorContext, right, left, vi) = 0, vi
dot_tilde_observe(::VarwisePriorContext, right, left, vi) = 0, vi

function acc_logp!(context::VarwisePriorContext, vn::Union{VarName,AbstractVector{<:VarName}}, logp)
    #sym = DynamicPPL.getsym(vn)  # leads to duplicates
    # if vn is a Vector leads to Symbol("VarName{:s, IndexLens{Tuple{Int64}}}[s[1], s[2]]")    
    sym = Symbol(vn) 
    context.logpriors[sym] = logp
    return (context)
end

"""
    varwise_logpriors(model::Model, chain::Chains; context)

Runs `model` on each sample in `chain` returning a tuple `(values, var_names)`
with var_names corresponding to symbols of the prior components, and values being 
array of shape `(num_samples, num_components, num_chains)`.

`context` specifies child context that handles computation of log-priors.

# Example
```julia; setup = :(using Distributions)
using DynamicPPL, Turing

@model function demo(x, ::Type{TV}=Vector{Float64}) where {TV}
        s ~ InverseGamma(2, 3)
        m = TV(undef, length(x))
        for i in eachindex(x)
            m[i] ~ Normal(0, √s)
        end
        x ~ MvNormal(m, √s)        
    end  

model = demo(randn(3), randn());

chain = sample(model, MH(), 10); 

lp = varwise_logpriors(model, chain)
# Can be used to construct a new Chains object
#lpc = MCMCChains(varwise_logpriors(model, chain)...)

# got a logdensity for each parameter prior 
(but fewer if used `.~` assignments, see below)
lp[2] == names(chain, :parameters)

# for each sample in the Chains object
size(lp[1])[[1,3]] == size(chain)[[1,3]]
```

# Broadcasting
Note that `m .~ Dist()` will treat `m` as a collection of
_independent_ prior rather than as a single prior,
but `varwise_logpriors` returns the single 
sum of log-likelihood of components of `m` only.
If one needs the log-density of the components, one needs to rewrite
the model with an explicit loop.
"""
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
