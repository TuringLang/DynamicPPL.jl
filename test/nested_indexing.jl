# HACK: Don't do this.
function Base.getindex(c::MCMCChains.Chains, vn::VarName)
    DynamicPPL.supports_varname_indexing(c) || error("Chains do not support indexing using $vn.")
    return c[:, c.info.varname_to_symbol[vn], :]
end
function Base.getindex(c::MCMCChains.Chains, vns::AbstractVector{<:VarName})
    DynamicPPL.supports_varname_indexing(c) || error("Chains do not support indexing using $vn.")
    return c[:, map(Base.Fix1(getindex, c.info.varname_to_symbol), vns), :]
end

@testset "generated_quantities" begin
    n = 10
    d = 2
    model = DynamicPPL.TestUtils.demo_lkjchol(d)
    xs = [model().x for _ in 1:n]

    # Extract varnames and values.
    vns_and_vals_xs = map(
        collect ∘ Base.Fix1(DynamicPPL.varname_and_value_leaves, @varname(x)), xs
    )
    vns = map(first, first(vns_and_vals_xs))
    vals = map(vns_and_vals_xs) do vns_and_vals
        map(last, vns_and_vals)
    end

    # Construct the chain.
    syms = map(Symbol, vns)
    vns_to_syms = OrderedDict(zip(vns, syms))

    chain = MCMCChains.Chains(
        permutedims(stack(vals)),
        syms;
        info = (varname_to_symbol = vns_to_syms,)
    )
    display(chain)

    # Test!
    results = generated_quantities(model, chain)
    for (x_true, result) in zip(xs, results)
        @test x_true.UL == result.x.UL
    end
end
