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
            $(esc(:__rng__)),
            $(esc(expr)),
            $(esc(:__varinfo__)),
            $(esc(:__sampler__)),
            $(DynamicPPL).PrefixContext{$(QuoteNode(prefix))}($(esc(:__context__)))
        )
    end
end
