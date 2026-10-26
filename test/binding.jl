module BindingTests

using Test
using DynamicPPL, Distributions, Random, LinearAlgebra
using StableRNGs: StableRNG
using AbstractPPL: @of, of

# Set DYNAMICPPL_EXHAUSTIVE_TESTS=true to run the complete binding occupancy enumeration.
const EXHAUSTIVE = get(ENV, "DYNAMICPPL_EXHAUSTIVE_TESTS", "false") == "true"

@model function oracle_argument(x, whole, entered)
    entered[] = true
    if whole
        x ~ MvNormal(zeros(2), I)
    else
        x[1] ~ Normal()
        x[2] ~ Normal()
    end
    return x
end
@model function oracle_local(whole, entered)
    x = zeros(2)
    entered[] = true
    if whole
        x ~ MvNormal(zeros(2), I)
    else
        x[1] ~ Normal()
        x[2] ~ Normal()
    end
    return x
end
@model oracle_parent(c) = a ~ to_submodel(c)
struct OracleInit <: DynamicPPL.AbstractInitStrategy end
function DynamicPPL.init(rng, vn, d::Distribution, ::OracleInit)
    value = d isa MultivariateDistribution ? fill(0.25, length(d)) : 0.25
    return DynamicPPL.TransformedValue(value, DynamicPPL.NoTransform())
end
const ROOTS = (@varname(x), @varname(a.x), @varname(a.a.x))
const PARTS = (
    (@varname(x[1]), @varname(x[2])),
    (@varname(a.x[1]), @varname(a.x[2])),
    (@varname(a.a.x[1]), @varname(a.a.x[2])),
)
selected(mask, i) = !iszero(mask & (1 << (i - 1)))
function bindmask(m, depth, mask, role, value)
    inputs = Tuple(PARTS[depth][i] => value + i for i in 1:2 if selected(mask, i))
    isempty(inputs) && return m
    return (role == 1 ? condition : fix)(m, inputs...)
end
function removemask(m, depth, mask, role, recursive)
    names = Tuple(PARTS[depth][i] for i in 1:2 if selected(mask, i))
    isempty(names) && return m
    f = role == 1 ? decondition : unfix
    return recursive ? f(m, DynamicPPL.Recursive(), names...) : f(m, names...)
end
function base_model(storage, whole, initial, entered)
    m = if storage == :argument
        oracle_argument([1.0, 2.0], whole, entered)
    elseif storage == :template
        used = condition(oracle_local(whole, entered), @of(x = of(Array, 2)); x=zeros(2))
        decondition(used, @varname(x))
    else
        condition(oracle_local(whole, entered); x=zeros(2))
    end
    for i in 1:2
        if storage != :argument || initial[i] != 1
            m = decondition(m, PARTS[1][i])
        end
    end
    return m
end
function predict(q, initial, firstop, secondop, removal, whole; removal_scope=q)
    roles = zeros(Int, q, 2)
    values = zeros(q, 2)
    for i in 1:2
        initial[i] == 0 && continue
        owner, role = (initial[i] + 1) ÷ 2, 1 + (initial[i] - 1) % 2
        roles[owner, i] = role
        values[owner, i] = Float64(i)
    end
    mask, role, recursive = removal
    for scope in 1:q
        if scope == removal_scope
            for owner in (recursive ? (1:scope) : (scope:scope)), i in 1:2
                selected(mask, i) && roles[owner, i] == role && (roles[owner, i] = 0)
            end
        end
        for (owner, subset, newrole, value) in (firstop, secondop)
            owner == scope || continue
            for i in 1:2
                selected(subset, i) || continue
                old = roles[scope, i]
                old != 0 && old != newrole && return (:call, string(PARTS[scope][i]))
                roles[scope, i] = newrole
                values[scope, i] = value + i
            end
        end
    end
    resultroles = zeros(Int, 2)
    resultvalues = fill(0.25, 2)
    for i in 1:2
        owners = [s for s in 1:q if roles[s, i] != 0]
        length(owners) > 1 && return (:entry, string(ROOTS[q]))
        if !isempty(owners)
            owner = only(owners)
            resultroles[i] = roles[owner, i]
            resultvalues[i] = values[owner, i]
        end
    end
    whole && resultroles[1] != resultroles[2] && return (:tilde, string(ROOTS[q]))
    density(x) = -(log(2pi) + x * x) / 2
    prior = sum((density(resultvalues[i]) for i in 1:2 if resultroles[i] == 0); init=0.0)
    likelihood = sum(
        (density(resultvalues[i]) for i in 1:2 if resultroles[i] == 1); init=0.0
    )
    listing = Tuple((roles[q, i], roles[q, i] == 0 ? 0.0 : values[q, i]) for i in 1:2)
    latent = if whole
        (all(iszero, resultroles) ? [string(ROOTS[q])] : String[])
    else
        [string(PARTS[q][i]) for i in 1:2 if resultroles[i] == 0]
    end
    return (:ok, (resultvalues, prior, likelihood, prior + likelihood, listing, latent))
end
oracle_inputs(x) = deepcopy(x)
oracle_inputs(::Base.RefValue{Bool}) = nothing
oracle_inputs(x::NamedTuple) = map(oracle_inputs, x)
function oracle_inputs(m::Model)
    return (oracle_inputs(m.args), deepcopy(conditioned(m)), deepcopy(fixed(m)))
end
function actual(storage, whole, q, initial, firstop, secondop, removal; removal_scope=q)
    entered = Ref(false)
    m = base_model(storage, whole, initial, entered)
    snapshots = []
    snapshot(m) = (deepcopy(conditioned(m)), deepcopy(fixed(m)), oracle_inputs(m.args))
    intact() = all(isequal(snapshot(model), before) for (model, before) in snapshots)
    push!(snapshots, (m, snapshot(m)))
    try
        for scope in 1:q
            scope > 1 && (m = oracle_parent(m))
            for i in 1:2
                initial[i] == 0 && continue
                owner, role = (initial[i] + 1) ÷ 2, 1 + (initial[i] - 1) % 2
                storage == :argument && owner == 1 && role == 1 && continue
                if owner == scope
                    push!(snapshots, (m, snapshot(m)))
                    m = bindmask(m, scope, 1 << (i - 1), role, 0.0)
                end
            end
            if scope == removal_scope
                push!(snapshots, (m, snapshot(m)))
                m = removemask(m, scope, removal...)
            end
            for (owner, subset, role, value) in (firstop, secondop)
                owner == scope || continue
                push!(snapshots, (m, snapshot(m)))
                m = bindmask(m, scope, subset, role, value)
            end
        end
    catch err
        err isa InterruptException && rethrow()
        intact() || return (:mutated, nothing)
        return (:call, (nameof(typeof(err)), sprint(showerror, err)))
    end
    push!(snapshots, (m, snapshot(m)))
    try
        before = snapshot(m)
        listing = Tuple(
            map(1:2) do i
                vn = PARTS[q][i]
                o, f = haskey(before[1], vn), haskey(before[2], vn)
                o && f && return (-1, 0.0)
                return if o
                    (1, before[1][vn])
                elseif f
                    (2, before[2][vn])
                else
                    (0, 0.0)
                end
            end,
        )
        value, vi = init!!(
            Xoshiro(1),
            m,
            VarInfo(
                RawValueAccumulator(false),
                LogPriorAccumulator(),
                LogLikelihoodAccumulator(),
            ),
            OracleInit(),
            UnlinkAll(),
        )
        intact() || return (:mutated, nothing)
        latent = sort!(string.(collect(keys(get_raw_values(vi)))))
        return (
            :ok,
            (
                value,
                getlogprior(vi),
                getloglikelihood(vi),
                getlogjoint(vi),
                listing,
                latent,
            ),
        )
    catch err
        err isa InterruptException && rethrow()
        intact() || return (:mutated, nothing)
        return (entered[] ? :tilde : :entry, (nameof(typeof(err)), sprint(showerror, err)))
    end
end
function agrees(expected, observed)
    expected[1] == observed[1] || return false
    if expected[1] != :ok
        return observed[2][1] === :ArgumentError && occursin(expected[2], observed[2][2])
    end
    e, o = expected[2], observed[2]
    return e[1] == o[1] &&
           all(isapprox(e[i], o[i]; atol=1e-12) for i in 2:4) &&
           e[5] == o[5] &&
           sort(e[6]) == o[6]
end
function run_oracle()
    failures, cases = 0, 0
    rng = StableRNG(1001)
    categories = Dict{Tuple{Symbol,Symbol},Int}()
    for storage in (:argument, :template, :held), whole in (false, true), q in 1:3
        startfail = failures
        for initial in Iterators.product(0:(2q), 0:(2q))
            for owner1 in 1:q, owner2 in 1:q, subset1 in 0:3, subset2 in 0:3
                for role1 in 1:2,
                    role2 in 1:2,
                    removal in 0:3,
                    removerole in 1:2,
                    recursive in (false, true),
                    offsets in ((2.0, 2.0), (2.0, 4.0), (4.0, 2.0))

                    mandatory =
                        initial == (0, 0) &&
                        owner1 == 1 &&
                        owner2 == q &&
                        subset1 == 3 &&
                        subset2 == 3 &&
                        role1 == role2 &&
                        removal == 3 &&
                        removerole == role1
                    !EXHAUSTIVE && !mandatory && rand(rng) > 0.00001 && continue
                    op1, op2 = (owner1, subset1, role1, offsets[1]),
                    (owner2, subset2, role2, offsets[2])
                    rem = (removal, removerole, recursive)
                    expected = predict(q, initial, op1, op2, rem, whole)
                    observed = actual(storage, whole, q, initial, op1, op2, rem)
                    cases += 1
                    valid = agrees(expected, observed)
                    if !valid
                        failures += 1
                        key = (expected[1], observed[1])
                        categories[key] = get(categories, key, 0) + 1
                        if failures <= 15
                            println(
                                "MISMATCH ",
                                (storage, whole, q, initial, op1, op2, rem),
                                " expected=",
                                expected,
                                " observed=",
                                observed,
                            )
                        end
                    end
                end
            end
        end
        println(
            "GROUP ",
            (storage, whole, q),
            " cases=",
            cases,
            " failures=",
            failures - startfail,
        )
        flush(stdout)
    end
    println(
        "TOTAL cases=",
        cases,
        " failures=",
        failures,
        " categories=",
        sort!(collect(categories); by=first),
    )
    @test failures == 0
end
function run_scope_and_batch_oracle()
    cases = failures = 0
    rng = StableRNG(1002)
    for storage in (:argument, :template, :held),
        whole in (false, true),
        q in 1:3,
        initial in Iterators.product(0:(2q), 0:(2q)),
        remover in 1:q,
        mask in 0:3,
        role in 1:2,
        recursive in (false, true),
        owner in 1:q,
        insertrole in 1:2

        mandatory =
            initial == (0, 0) &&
            remover == 1 &&
            owner == q &&
            mask == 3 &&
            role == insertrole &&
            recursive
        !EXHAUSTIVE && !mandatory && rand(rng) > 0.00001 && continue
        op = (owner, 3, insertrole, 2.0)
        noop = (owner, 0, insertrole, 2.0)
        removal = (mask, role, recursive)
        expected = predict(q, initial, op, noop, removal, whole; removal_scope=remover)
        observed = actual(
            storage, whole, q, initial, op, noop, removal; removal_scope=remover
        )
        cases += 1
        if !agrees(expected, observed)
            failures += 1
            failures <= 15 && println(
                "SCOPE MISMATCH ",
                (storage, whole, q, initial, remover, removal, op),
                " expected=",
                expected,
                " observed=",
                observed,
            )
        end
    end
    for storage in (:argument, :template, :held),
        whole in (false, true),
        q in 1:3,
        role in 1:2,
        position in 1:2,
        second in (3.0, 4.0)

        m = base_model(storage, whole, (0, 0), Ref(false))
        for scope in 2:q
            m = oracle_parent(m)
        end
        before = oracle_inputs(m)
        vn = PARTS[q][position]
        result = try
            (role == 1 ? condition : fix)(m, vn => 3.0, vn => second)
            false
        catch err
            err isa InterruptException && rethrow()
            err isa ArgumentError && occursin(string(vn), sprint(showerror, err))
        end
        cases += 1
        if !result || !isequal(before, oracle_inputs(m))
            failures += 1
            failures <= 15 &&
                println("BATCH MISMATCH ", (storage, whole, q, role, position, second))
        end
    end
    println("SCOPE_AND_BATCH cases=", cases, " failures=", failures)
    @test failures == 0
end
@testset "binding occupancy across model scopes" begin
    elapsed = @elapsed begin
        run_oracle()
        run_scope_and_batch_oracle()
    end
    println("Binding oracle seconds=", elapsed)
end

end
