macro logprob_str(str)
    return :(@warn(
        "`@logprob` is deprecated. Use `loglikelihood`, `logprior`, or `logjoint` instead."
    ))
end

macro prob_str(str)
    return :(@warn(
        "`@prob` is deprecated. Use `loglikelihood`, `logprior`, or `logjoint` instead."
    ))
end
