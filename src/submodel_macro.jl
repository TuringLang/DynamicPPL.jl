macro submodel(expr)
    return quote
        _evaluate(
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
        _evaluate(
            $(esc(:__rng__)),
            $(esc(expr)),
            $(esc(:__varinfo__)),
            $(esc(:__sampler__)),
            PrefixContext{$(esc(Meta.quot(prefix)))}($(esc(:__context__)))
        )
    end
end
