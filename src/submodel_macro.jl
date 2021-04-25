macro submodel(expr)
    return quote
        $(DynamicPPL)._evaluate(
            $(esc(:__rng__)),
            $(esc(expr)),
            $(esc(:__varinfo__)),
            $(esc(:__sampler__)),
            $(esc(:__context__))
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
