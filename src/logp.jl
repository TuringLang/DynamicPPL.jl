"""
    logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)

Return an array of log priors evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> using Random, MCMCChains

julia> Random.seed!(111)
MersenneTwister(111)

julia> val = rand(500, 2, 3)
500×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.390386  0.0837452
 0.485358  0.637198
 0.519265  0.575001
 0.057931  0.606497
 0.845293  0.867838
 0.264531  0.646586
 0.938287  0.610489
 ⋮
 0.570775  0.36348
 0.65202   0.371192
 0.579922  0.57587
 0.929339  0.968619
 0.997855  0.177522
 0.726988  0.0112906
 0.411814  0.450745

[:, :, 2] =
 0.14221   0.816536
 0.635809  0.422885
 0.359449  0.0230222
 0.868051  0.313322
 0.718046  0.15864
 0.703807  0.703968
 0.215787  0.24148
 ⋮
 0.395223  0.461259
 0.156243  0.281266
 0.17801   0.642005
 0.399439  0.477756
 0.301772  0.0444831
 0.514639  0.781145
 0.994754  0.932154

[:, :, 3] =
 0.837316   0.163184
 0.386275   0.37846
 0.751263   0.965999
 0.667769   0.0963822
 0.0440691  0.71489
 0.521811   0.0560717
 0.540227   0.204813
 ⋮
 0.28595    0.0897536
 0.769292   0.328548
 0.984564   0.396496
 0.142841   0.664338
 0.185144   0.1697
 0.318288   0.91384
 0.683415   0.606488

julia> chain = Chains(val, [:s, :m]) # construct a chain of samples using MCMCChains
Chains MCMC chain (500×2×3 Array{Float64, 3}):

Iterations        = 1:1:500
Number of chains  = 3
Samples per chain = 500
parameters        = s, m

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64

           s    0.4950    0.2858     0.0074    0.0081   1471.9090    0.9987
           m    0.4896    0.2856     0.0074    0.0077   1413.4976    0.9984

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64

           s    0.0370    0.2549    0.4820    0.7424    0.9769
           m    0.0241    0.2415    0.4910    0.7389    0.9634


julia> logprior(demo_model(x), chain)
500×3 Matrix{Float64}:
  -3.12323  -15.3349    -1.69905
  -2.79095   -1.99575   -3.34438
  -2.52378   -3.48741   -2.33505
 -43.7125    -1.73901   -1.8079
  -2.12802   -1.75797  -61.6682
  -6.19847   -2.10693   -2.19736
  -1.89469   -7.39229   -2.15858
   ⋮
  -2.1308    -3.33246   -4.84536
  -1.93158  -11.6785    -1.77357
  -2.27373  -10.6917    -1.79414
  -2.19811   -3.30603  -14.4578
  -1.73644   -4.47298   -9.09991
  -1.73246   -2.81886   -5.45221
  -3.14809   -2.15587   -2.04826
```   
"""
function logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end


"""
    loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)

Return an array of log likelihoods evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> using Random, MCMCChains

julia> Random.seed!(111)
MersenneTwister(111)

julia> val = rand(500, 2, 3)
500×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.390386  0.0837452
 0.485358  0.637198
 0.519265  0.575001
 0.057931  0.606497
 0.845293  0.867838
 0.264531  0.646586
 0.938287  0.610489
 ⋮
 0.570775  0.36348
 0.65202   0.371192
 0.579922  0.57587
 0.929339  0.968619
 0.997855  0.177522
 0.726988  0.0112906
 0.411814  0.450745

[:, :, 2] =
 0.14221   0.816536
 0.635809  0.422885
 0.359449  0.0230222
 0.868051  0.313322
 0.718046  0.15864
 0.703807  0.703968
 0.215787  0.24148
 ⋮
 0.395223  0.461259
 0.156243  0.281266
 0.17801   0.642005
 0.399439  0.477756
 0.301772  0.0444831
 0.514639  0.781145
 0.994754  0.932154

[:, :, 3] =
 0.837316   0.163184
 0.386275   0.37846
 0.751263   0.965999
 0.667769   0.0963822
 0.0440691  0.71489
 0.521811   0.0560717
 0.540227   0.204813
 ⋮
 0.28595    0.0897536
 0.769292   0.328548
 0.984564   0.396496
 0.142841   0.664338
 0.185144   0.1697
 0.318288   0.91384
 0.683415   0.606488

julia> chain = Chains(val, [:s, :m])
Chains MCMC chain (500×2×3 Array{Float64, 3}):

Iterations        = 1:1:500
Number of chains  = 3
Samples per chain = 500
parameters        = s, m

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64

           s    0.4950    0.2858     0.0074    0.0081   1471.9090    0.9987
           m    0.4896    0.2856     0.0074    0.0077   1413.4976    0.9984

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64

           s    0.0370    0.2549    0.4820    0.7424    0.9769
           m    0.0241    0.2415    0.4910    0.7389    0.9634


julia> loglikelihoods(demo_model(x), chain)
500×3 Matrix{Float64}:
 -2710.79  -3362.82   -1802.48
 -1657.18  -1687.59   -2149.22
 -1665.32  -3022.72   -1403.17
 -8903.99  -1655.76   -2022.45
 -1401.05  -1892.41  -10912.7
 -2259.29  -1471.2    -2335.01
 -1466.4   -3638.66   -2055.84
     ⋮
 -1809.32  -2002.24   -3362.31
 -1722.59  -4619.92   -1685.76
 -1607.6   -3044.81   -1569.36
 -1389.27  -1969.64   -3617.69
 -1721.95  -3367.02   -4443.91
 -2068.38  -1544.86   -1835.49
 -1974.14  -1391.46   -1526.21
```  
"""
function loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict)
    end
end


"""
    logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)

Return an array of log posteriors evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> using Random, MCMCChains

julia> Random.seed!(111)
MersenneTwister(111)

julia> val = rand(500, 2, 3)
500×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.390386  0.0837452
 0.485358  0.637198
 0.519265  0.575001
 0.057931  0.606497
 0.845293  0.867838
 0.264531  0.646586
 0.938287  0.610489
 ⋮
 0.570775  0.36348
 0.65202   0.371192
 0.579922  0.57587
 0.929339  0.968619
 0.997855  0.177522
 0.726988  0.0112906
 0.411814  0.450745

[:, :, 2] =
 0.14221   0.816536
 0.635809  0.422885
 0.359449  0.0230222
 0.868051  0.313322
 0.718046  0.15864
 0.703807  0.703968
 0.215787  0.24148
 ⋮
 0.395223  0.461259
 0.156243  0.281266
 0.17801   0.642005
 0.399439  0.477756
 0.301772  0.0444831
 0.514639  0.781145
 0.994754  0.932154

[:, :, 3] =
 0.837316   0.163184
 0.386275   0.37846
 0.751263   0.965999
 0.667769   0.0963822
 0.0440691  0.71489
 0.521811   0.0560717
 0.540227   0.204813
 ⋮
 0.28595    0.0897536
 0.769292   0.328548
 0.984564   0.396496
 0.142841   0.664338
 0.185144   0.1697
 0.318288   0.91384
 0.683415   0.606488

julia> chain = Chains(val, [:s, :m]) # construct a chain of samples using MCMCChains
Chains MCMC chain (500×2×3 Array{Float64, 3}):

Iterations        = 1:1:500
Number of chains  = 3
Samples per chain = 500
parameters        = s, m

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64

           s    0.4950    0.2858     0.0074    0.0081   1471.9090    0.9987
           m    0.4896    0.2856     0.0074    0.0077   1413.4976    0.9984

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64

           s    0.0370    0.2549    0.4820    0.7424    0.9769
           m    0.0241    0.2415    0.4910    0.7389    0.9634


julia> logjoint(demo_model(x), chain)
500×3 Matrix{Float64}:
 -2713.91  -3378.15   -1804.18
 -1659.97  -1689.59   -2152.57
 -1667.85  -3026.21   -1405.5
 -8947.7   -1657.5    -2024.26
 -1403.18  -1894.17  -10974.3
 -2265.48  -1473.3    -2337.2
 -1468.29  -3646.05   -2058.0
     ⋮
 -1811.45  -2005.57   -3367.15
 -1724.52  -4631.6    -1687.54
 -1609.87  -3055.5    -1571.16
 -1391.47  -1972.95   -3632.15
 -1723.69  -3371.49   -4453.01
 -2070.11  -1547.68   -1840.94
 -1977.29  -1393.61   -1528.26
```   
"""
function logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(Iterators.product(1:size(chain, 1), 1:size(chain, 3))) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi) for
            vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict) +
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end