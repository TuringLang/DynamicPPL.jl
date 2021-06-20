macro submodel(expr)
    args_tilde = getargs_tilde(expr)
    return if args_tilde === nothing
        # In this case we only want to get the `__varinfo__`.
        quote
            $(esc(:_)), $(esc(:__varinfo__)) = _evaluate_with_varinfo($(esc(expr)), $(esc(:__varinfo__)), $(esc(:__context__)))
        end
    else
        # Here we also want the return-variable.
        L, R = args_tilde
        quote
            $(esc(L)), $(esc(:__varinfo__)) = _evaluate_with_varinfo($(esc(R)), $(esc(:__varinfo__)), $(esc(:__context__)))
        end
    end
end

macro submodel(prefix, expr)
    ctx = :(PrefixContext{$(esc(Meta.quot(prefix)))}($(esc(:__context__))))

    args_tilde = getargs_tilde(expr)
    return if args_tilde === nothing
        # In this case we only want to get the `__varinfo__`.
        quote
            $(esc(:_)), $(esc(:__varinfo__)) = _evaluate_with_varinfo($(esc(expr)), $(esc(:__varinfo__)), $(ctx))
        end
    else
        # Here we also want the return-variable.
        L, R = args_tilde
        quote
            $(esc(L)), $(esc(:__varinfo__)) = _evaluate_with_varinfo($(esc(R)), $(esc(:__varinfo__)), $(ctx))
        end
    end
end
