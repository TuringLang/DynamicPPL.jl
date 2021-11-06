"""
    @submodel model

Run a Turing `model` nested inside of a Turing model.

The return value can be assigned to a variable.

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
    return submodel(expr)
end

"""
    @submodel prefix model

Run a Turing `model` nested inside of a Turing model and add "`prefix`." as a prefix
to all random variables inside of the `model`.

The prefix makes it possible to run the same Turing model multiple times while
keeping track of all random variables correctly.

The return value can be assigned to a variable.

# Examples

```jldoctest submodelprefix; setup=:(using Distributions)
julia> @model function demo1(x)
           x ~ Normal()
           return 1 + abs(x)
       end;

julia> @model function demo2(x, y, z)
            a = @submodel sub1 demo1(x)
            b = @submodel sub2 demo1(y)
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
"""
macro submodel(prefix, expr)
    ctx = :(PrefixContext{$(esc(Meta.quot(prefix)))}($(esc(:__context__))))
    return submodel(expr, ctx)
end

function submodel(expr, ctx=esc(:__context__))
    args_assign = getargs_assignment(expr)
    return if args_assign === nothing
        # In this case we only want to get the `__varinfo__`.
        quote
            $(esc(:_)), $(esc(:__varinfo__)) = _evaluate(
                $(esc(expr)), $(esc(:__varinfo__)), $(ctx)
            )
        end
    else
        # Here we also want the return-variable.
        # TODO: Should we prefix by `L` by default?
        L, R = args_assign
        quote
            $(esc(L)), $(esc(:__varinfo__)) = _evaluate(
                $(esc(R)), $(esc(:__varinfo__)), $(ctx)
            )
        end
    end
end
