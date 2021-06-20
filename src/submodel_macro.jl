"""
    @submodel x ~ model(args...)
    @submodel prefix x ~ model(args...)

Treats `model` as a distribution, where `x` is the return-value of `model`.

If `prefix` is specified, then variables sampled within `model` will be
prefixed by `prefix`. This is useful if you have variables of same names in
several models used together.
"""
macro submodel(expr)
    args_tilde = getargs_tilde(expr)
    return if args_tilde === nothing
        # In this case we only want to get the `__varinfo__`.
        quote
            $(esc(:_)), $(esc(:__varinfo__)) = _evaluate_with_varinfo(
                $(esc(expr)), $(esc(:__varinfo__)), $(esc(:__context__))
            )
        end
    else
        # Here we also want the return-variable.
        L, R = args_tilde
        quote
            $(esc(L)), $(esc(:__varinfo__)) = _evaluate_with_varinfo(
                $(esc(R)), $(esc(:__varinfo__)), $(esc(:__context__))
            )
        end
    end
end

macro submodel(prefix, expr)
    ctx = :(PrefixContext{$(esc(Meta.quot(prefix)))}($(esc(:__context__))))

    args_tilde = getargs_tilde(expr)
    return if args_tilde === nothing
        # In this case we only want to get the `__varinfo__`.
        quote
            $(esc(:_)), $(esc(:__varinfo__)) = _evaluate_with_varinfo(
                $(esc(expr)), $(esc(:__varinfo__)), $(ctx)
            )
        end
    else
        # Here we also want the return-variable.
        L, R = args_tilde
        quote
            $(esc(L)), $(esc(:__varinfo__)) = _evaluate_with_varinfo(
                $(esc(R)), $(esc(:__varinfo__)), $(ctx)
            )
        end
    end
end
