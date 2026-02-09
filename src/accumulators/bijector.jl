struct BijectorAccumulator <: AbstractAccumulator
    bijectors::Vector{Any}
    sizes::Vector{Int}
end

BijectorAccumulator() = BijectorAccumulator(Bijectors.Bijector[], UnitRange{Int}[])

function Base.:(==)(acc1::BijectorAccumulator, acc2::BijectorAccumulator)
    return (acc1.bijectors == acc2.bijectors && acc1.sizes == acc2.sizes)
end

function Base.copy(acc::BijectorAccumulator)
    return BijectorAccumulator(copy(acc.bijectors), copy(acc.sizes))
end

accumulator_name(::Type{<:BijectorAccumulator}) = :Bijector

function _zero(acc::BijectorAccumulator)
    return BijectorAccumulator(empty(acc.bijectors), empty(acc.sizes))
end
reset(acc::BijectorAccumulator) = _zero(acc)
split(acc::BijectorAccumulator) = _zero(acc)
function combine(acc1::BijectorAccumulator, acc2::BijectorAccumulator)
    return BijectorAccumulator(
        vcat(acc1.bijectors, acc2.bijectors), vcat(acc1.sizes, acc2.sizes)
    )
end

function accumulate_assume!!(
    acc::BijectorAccumulator, val, tval, logjac, vn, right, template
)
    bijector = _compose_no_identity(
        to_linked_vec_transform(right), from_vec_transform(right)
    )
    push!(acc.bijectors, bijector)
    push!(acc.sizes, prod(output_size(to_vec_transform(right), right); init=1))
    return acc
end

accumulate_observe!!(acc::BijectorAccumulator, right, left, vn) = acc

"""
    bijector(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())

Returns a `Stacked <: Bijector` which maps from constrained to unconstrained space.

The input to the bijector is a vector of values for the whole model, like the input to
`unflatten!!`. These are in constrained space, i.e., respecting variable constraints.
The output is a vector of unconstrained values.

`init_strategy` is passed to `DynamicPPL.init!!` to determine what values the model is
evaluated with. This may affect the results if the prior distributions or constraints of
variables are dependent on other variables.
"""
function Bijectors.bijector(
    model::DynamicPPL.Model, init_strategy::AbstractInitStrategy=InitFromPrior()
)
    vi = OnlyAccsVarInfo((BijectorAccumulator(),))
    vi = last(DynamicPPL.init!!(model, vi, init_strategy, UnlinkAll()))
    acc = getacc(vi, Val(:Bijector))
    ranges = foldl(acc.sizes; init=UnitRange{Int}[]) do cumulant, sz
        last_index = length(cumulant) > 0 ? last(cumulant).stop : 0
        push!(cumulant, (last_index + 1):(last_index + sz))
        return cumulant
    end
    return Bijectors.Stacked(acc.bijectors, ranges)
end
