macro submodel(expr)
    return quote
        $(DynamicPPL)._evaluate(
            $(esc(:_rng)),
            $(esc(expr)),
            $(esc(:_varinfo)),
            $(esc(:_sampler)),
            $(esc(:_context))
        )
    end
end

macro submodel(prefix, expr)
    return quote
        $(DynamicPPL)._evaluate(
            $(esc(:_rng)),
            $(esc(expr)),
            $(esc(:_varinfo)),
            $(esc(:_sampler)),
            $(PrefixContext){$(esc(prefix))}($(esc(:_context)))
        )
    end
end
