"""
    @submodel model
    @submodel ... = model

Run a Turing `model` nested inside of a Turing model.

# Examples

```jldoctest submodel; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y)
            @submodel a = demo1(x)
            return y ~ Uniform(0, a)
       end;
```

When we sample from the model `demo2(missing, 0.4)` random variable `x` will be sampled:
```jldoctest submodel
julia> vi = VarInfo(demo2(missing, 0.4));

julia> @varname(x) in keys(vi)
true
```

Variable `a` is not tracked since it can be computed from the random variable `x` that was
tracked when running `demo1`:
```jldoctest submodel
julia> @varname(a) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel
julia> x = vi[@varname(x)];

julia> getlogp(vi) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```
"""
macro submodel(expr)
    return submodel(:(prefix = false), expr)
end

"""
    @submodel prefix=... model
    @submodel prefix=... ... = model

Run a Turing `model` nested inside of a Turing model and add "`prefix`." as a prefix
to all random variables inside of the `model`.

Valid expressions for `prefix=...` are:
- `prefix=false`: no prefix is used.
- `prefix=true`: _attempt_ to automatically determine the prefix from the left-hand side
  `... = model` by first converting into a `VarName`, and then calling `Symbol` on this.
- `prefix=expression`: results in the prefix `Symbol(expression)`.

The prefix makes it possible to run the same Turing model multiple times while
keeping track of all random variables correctly.

# Examples
## Example models
```jldoctest submodelprefix; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y, z)
            @submodel prefix="sub1" a = demo1(x)
            @submodel prefix="sub2" b = demo1(y)
            return z ~ Uniform(-a, b)
       end;
```

When we sample from the model `demo2(missing, missing, 0.4)` random variables `sub1.x` and
`sub2.x` will be sampled:
```jldoctest submodelprefix
julia> vi = VarInfo(demo2(missing, missing, 0.4));

julia> @varname(var"sub1.x") in keys(vi)
true

julia> @varname(var"sub2.x") in keys(vi)
true
```

Variables `a` and `b` are not tracked since they can be computed from the random variables `sub1.x` and
`sub2.x` that were tracked when running `demo1`:
```jldoctest submodelprefix
julia> @varname(a) in keys(vi)
false

julia> @varname(b) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodelprefix
julia> sub1_x = vi[@varname(var"sub1.x")];

julia> sub2_x = vi[@varname(var"sub2.x")];

julia> logprior = logpdf(Normal(), sub1_x) + logpdf(Normal(), sub2_x);

julia> loglikelihood = logpdf(Uniform(-1 - abs(sub1_x), 1 + abs(sub2_x)), 0.4);

julia> getlogp(vi) ≈ logprior + loglikelihood
true
```

## Different ways of setting the prefix
```jldoctest submodel-prefix-alternatives; setup=:(using DynamicPPL, Distributions)
julia> @model inner() = x ~ Normal()
inner (generic function with 2 methods)

julia> # When `prefix` is unspecified, no prefix is used.
       @model submodel_noprefix() = @submodel a = inner()
submodel_noprefix (generic function with 2 methods)

julia> @varname(x) in keys(VarInfo(submodel_noprefix()))
true

julia> # Explicitely don't use any prefix.
       @model submodel_prefix_false() = @submodel prefix=false a = inner()
submodel_prefix_false (generic function with 2 methods)

julia> @varname(x) in keys(VarInfo(submodel_prefix_false()))
true

julia> # Automatically determined from `a`.
       @model submodel_prefix_true() = @submodel prefix=true a = inner()
submodel_prefix_true (generic function with 2 methods)

julia> @varname(var"a.x") in keys(VarInfo(submodel_prefix_true()))
true

julia> # Using a static string.
       @model submodel_prefix_string() = @submodel prefix="my prefix" a = inner()
submodel_prefix_string (generic function with 2 methods)

julia> @varname(var"my prefix.x") in keys(VarInfo(submodel_prefix_string()))
true

julia> # Using string interpolation.
       @model submodel_prefix_interpolation() = @submodel prefix="\$(nameof(inner()))" a = inner()
submodel_prefix_interpolation (generic function with 2 methods)

julia> @varname(var"inner.x") in keys(VarInfo(submodel_prefix_interpolation()))
true

julia> # Or using some arbitrary expression.
       @model submodel_prefix_expr() = @submodel prefix=1 + 2 a = inner()
submodel_prefix_expr (generic function with 2 methods)

julia> @varname(var"3.x") in keys(VarInfo(submodel_prefix_expr()))
true

julia> # (×) Automatic prefixing without a left-hand side expression does not work!
       @model submodel_prefix_error() = @submodel prefix=true inner()
ERROR: LoadError: cannot automatically prefix with no left-hand side
[...]
```

# Notes
- The choice `prefix=expression` means that the prefixing will incur a runtime cost.
  This is also the case for `prefix=true`, depending on whether the expression on the
  the right-hand side of `... = model` requires runtime-information or not, e.g.
  `x = model` will result in the _static_ prefix `x`, while `x[i] = model` will be
  resolved at runtime.
"""
macro submodel(prefix_expr, expr)
    return submodel(prefix_expr, expr, esc(:__context__))
end

# Automatic prefixing.
function prefix_submodel_context(prefix::Bool, left::Symbol, ctx)
    return prefix ? prefix_submodel_context(left, ctx) : ctx
end

function prefix_submodel_context(prefix::Bool, left::Expr, ctx)
    return prefix ? prefix_submodel_context(varname(left), ctx) : ctx
end

# Manual prefixing.
prefix_submodel_context(prefix, left, ctx) = prefix_submodel_context(prefix, ctx)
function prefix_submodel_context(prefix, ctx)
    # E.g. `prefix="asd[$i]"` or `prefix=asd` with `asd` to be evaluated.
    return :($(PrefixContext){$(Symbol)($(esc(prefix)))}($ctx))
end

function prefix_submodel_context(prefix::Union{AbstractString,Symbol}, ctx)
    # E.g. `prefix="asd"`.
    return :($(PrefixContext){$(esc(Meta.quot(Symbol(prefix))))}($ctx))
end

function prefix_submodel_context(prefix::Bool, ctx)
    if prefix
        error("cannot automatically prefix with no left-hand side")
    end

    return ctx
end

function submodel(prefix_expr, expr, ctx=esc(:__context__))
    prefix_left, prefix = getargs_assignment(prefix_expr)
    if prefix_left !== :prefix
        error("$(prefix_left) is not a valid kwarg")
    end

    # The user expects `@submodel ...` to return the
    # return-value of the `...`, hence we need to capture
    # the return-value and handle it correctly.
    @gensym retval

    # `prefix=false` => don't prefix, i.e. do nothing to `ctx`.
    # `prefix=true` => automatically determine prefix.
    # `prefix=...` => use it.
    args_assign = getargs_assignment(expr)
    return if args_assign === nothing
        ctx = prefix_submodel_context(prefix, ctx)
        quote
            $retval, $(esc(:__varinfo__)) = $(_evaluate!!)(
                $(esc(expr)), $(esc(:__varinfo__)), $(ctx)
            )
            $retval
        end
    else
        L, R = args_assign
        # Now that we have `L` and `R`, we can prefix automagically.
        try
            ctx = prefix_submodel_context(prefix, L, ctx)
        catch e
            error(
                "failed to determine prefix from $(L); please specify prefix using the `@submodel prefix=\"your prefix\" ...` syntax",
            )
        end
        quote
            $retval, $(esc(:__varinfo__)) = $(_evaluate!!)(
                $(esc(R)), $(esc(:__varinfo__)), $(ctx)
            )
            $(esc(L)) = $retval
        end
    end
end

"""
    @returned_quantities model

Run `model` nested inside of another model and return the return-values of the `model`.

!!! warning
    It's generally recommended to use [`prefix(::Model, x)`](@ref) or
    [`@prefix(model, prefix_expr)`](@ref) in combination with `@returned_quantities`
    to ensure that the variables in `model` are unique and do not clash with other variables in the
    parent model or in other submodels.

# Examples

## Simple example
```jldoctest submodel-returned-quantities; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y)
            a = @returned_quantities(demo1(x))
            return y ~ Uniform(0, a)
       end;
```

When we sample from the model `demo2(missing, 0.4)` random variable `x` will be sampled:
```jldoctest submodel-returned-quantities
julia> vi = VarInfo(demo2(missing, 0.4));

julia> @varname(x) in keys(vi)
true
```

Variable `a` is not tracked since it can be computed from the random variable `x` that was
tracked when running `demo1`:
```jldoctest submodel-returned-quantities
julia> @varname(a) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-returned-quantities
julia> x = vi[@varname(x)];

julia> getlogp(vi) ≈ logpdf(Normal(), x) + logpdf(Uniform(0, 1 + abs(x)), 0.4)
true
```

## With prefixing
```jldoctest submodel-returned-quantities-prefix; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y, z)
            a = @returned_quantities prefix(demo1(x), Val{:sub1}())
            b = @returned_quantities prefix(demo1(y), Val{:sub2}())
            return z ~ Uniform(-a, b)
       end;
```

When we sample from the model `demo2(missing, missing, 0.4)` random variables `sub1.x` and
`sub2.x` will be sampled:
```jldoctest submodel-returned-quantities-prefix
julia> vi = VarInfo(demo2(missing, missing, 0.4));

julia> @varname(var"sub1.x") in keys(vi)
true

julia> @varname(var"sub2.x") in keys(vi)
true
```

Variables `a` and `b` are not tracked since they can be computed from the random variables `sub1.x` and
`sub2.x` that were tracked when running `demo1`:
```jldoctest submodel-returned-quantities-prefix
julia> @varname(a) in keys(vi)
false

julia> @varname(b) in keys(vi)
false
```

We can check that the log joint probability of the model accumulated in `vi` is correct:

```jldoctest submodel-returned-quantities-prefix
julia> sub1_x = vi[@varname(var"sub1.x")];

julia> sub2_x = vi[@varname(var"sub2.x")];

julia> logprior = logpdf(Normal(), sub1_x) + logpdf(Normal(), sub2_x);

julia> loglikelihood = logpdf(Uniform(-1 - abs(sub1_x), 1 + abs(sub2_x)), 0.4);

julia> getlogp(vi) ≈ logprior + loglikelihood
true
```

## Different ways of setting the prefix
```jldoctest submodel-returned-quantities-prefix-alts; setup=:(using DynamicPPL, Distributions)
julia> @model inner() = x ~ Normal()
inner (generic function with 2 methods)

julia> # When `prefix` is unspecified, no prefix is used.
       @model submodel_noprefix() = a = @returned_quantities inner()
submodel_noprefix (generic function with 2 methods)

julia> @varname(x) in keys(VarInfo(submodel_noprefix()))
true

julia> # Explicitely don't use any prefix.
       @model submodel_prefix_false() = a = @returned_quantities prefix=false inner()
submodel_prefix_false (generic function with 2 methods)

julia> @varname(x) in keys(VarInfo(submodel_prefix_false()))
true

julia> # Using a static string.
       @model submodel_prefix_string() = a = @returned_quantities prefix="my prefix" inner()
submodel_prefix_string (generic function with 2 methods)

julia> @varname(var"my prefix.x") in keys(VarInfo(submodel_prefix_string()))
true

julia> # Using string interpolation.
       @model submodel_prefix_interpolation() = a = @returned_quantities prefix="\$(nameof(inner()))" inner()
submodel_prefix_interpolation (generic function with 2 methods)

julia> @varname(var"inner.x") in keys(VarInfo(submodel_prefix_interpolation()))
true

julia> # Or using some arbitrary expression.
       @model submodel_prefix_expr() = a = @returned_quantities prefix=1 + 2 inner()
submodel_prefix_expr (generic function with 2 methods)

julia> @varname(var"3.x") in keys(VarInfo(submodel_prefix_expr()))
true
```
"""
macro returned_quantities(expr)
    return returned_quantities_expr(:(prefix = false), expr)
end

macro returned_quantities(prefix_expr, expr)
    return returned_quantities_expr(prefix_expr, expr)
end

"""
    @returned_quantities_expr model

Returns an expression that captures the return-values of a model in addition to the varinfo.
"""
function returned_quantities_expr(prefix_expr, expr, ctx=esc(:__context__))
    prefix_left, prefix = getargs_assignment(prefix_expr)
    if prefix_left !== :prefix
        error("$(prefix_left) is not a valid kwarg")
    end

    # The user expects `@submodel ...` to return the
    # return-value of the `...`, hence we need to capture
    # the return-value and handle it correctly.
    @gensym retval

    # Prefix.
    if prefix !== nothing
        ctx = prefix_submodel_context(prefix, ctx)
    end
    return quote
        # Evaluate the model and capture the return values + varinfo.
        $retval, $(esc(:__varinfo__)) = $(_evaluate!!)(
            $(esc(expr)), $(esc(:__varinfo__)), $(ctx)
        )

        # Return the return-value of the model.
        $retval
    end
end
