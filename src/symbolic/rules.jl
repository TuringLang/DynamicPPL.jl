using Bijectors: Bijectors
using Symbolics: Symbolics
using Symbolics.SymbolicUtils

Symbolics.@register Bijectors.logpdf_with_trans(dist, r, istrans)

# Some predicates
isdist(d) = (d isa Type) && (d <: Distribution)
islogpdf(f::Function) = f === Distributions.logpdf
islogpdf(x) = false

# HACK: Apparently this is needed for disambiguiation.
# TODO: Open issue.
function Symbolics.:<ₑ(a::Real, b::Symbolics.Num)
    return Symbolics.:<ₑ(Symbolics.value(a), Symbolics.value(b))
end
function Symbolics.:<ₑ(a::Symbolics.Num, b::Real)
    return Symbolics.:<ₑ(Symbolics.value(a), Symbolics.value(b))
end

#############
### Rules ###
#############
# HACK: We'll wrap rewriters to add back `Num`. This way we can get jacobians and whatnot at then end.
const rmnum_rule = @rule (~x) => Symbolics.value(~x)
const addnum_rule = @rule (~x) => Symbolics.Num(~x)

# In the case where we want to work directly with the `x ~ Distribution` statements, the following rules can be useful:
const logpdf_rule = @rule (~x ~ ~d) =>
    Distributions.logpdf(Symbolics.Num(~d), Symbolics.Num(~x));
const rand_rule = @rule (~x ~ ~d) => Distributions.rand(Symbolics.Num(~d))

# We don't want to trace into `Bijectors.logpdf_with_trans`, so we just replace it with `logpdf`.
islogpdf_with_trans(f::Function) = f === Bijectors.logpdf_with_trans
islogpdf_with_trans(x) = false
const logpdf_with_trans_rule = @rule (~f::islogpdf_with_trans)(~dist, ~x, ~istrans) =>
    logpdf(~dist, ~x)

# Attempt to expand `logpdf` to get analytical expressions.
# The idea is that `getlogpdf(d, args)` should return a method of the following signature:
#
#    f(args..., x)
#
# which returns the logpdf.
# HACK: this is very hacky but you get the idea
import Distributions: StatsFuns
function getlogpdf(d, args)
    replacements = Dict(:Normal => StatsFuns.normlogpdf, :Gamma => StatsFuns.gammalogpdf)

    dsym = Symbol(d)
    if haskey(replacements, dsym)
        return replacements[dsym]
    else
        return d
    end
end

const analytic_rule = @rule (~f::islogpdf)((~d::isdist)(~~args), ~x) =>
    getlogpdf(~d, ~~args)(map(Symbolics.Num, (~~args))..., Symbolics.Num(~x))

#################
### Rewriters ###
#################
# TODO: these should probably be instantiated when needed, rather than here.
const analytic_rw = Rewriters.Postwalk(
    Rewriters.Chain((
        rmnum_rule,             # 0. Remove `Num` so we're only working stuff from `SymbolicUtils.jl`.
        logpdf_with_trans_rule, # 1. Replace `logpdf_with_trans` with `logpdf`.
        analytic_rule,          # 2. Attempt to replace `logpdf` with analytic expression.
    ))
)

# So we add back `Num` to all terms to allow differentiation.
const rmnum_rw = Rewriters.Postwalk(Rewriters.PassThrough(rmnum_rule))
const addnum_rw = Rewriters.Postwalk(Rewriters.PassThrough(addnum_rule))
