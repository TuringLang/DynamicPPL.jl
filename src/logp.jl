"""
    logprior(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log priors evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> @model function demo_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
    end
end
demo_model (generic function with 2 methods)

julia> using StableRNGs, MCMCChains

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3)
10×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.530608  0.725495
 0.680607  0.0657651
 0.301843  0.697142
 0.696529  0.434764
 0.677034  0.15186
 0.579365  0.57197
 0.662948  0.998378
 0.187804  0.7701
 0.871568  0.797581
 0.111344  0.750877

[:, :, 2] =
 0.355385   0.778479
 0.195597   0.297894
 0.827821   0.760928
 0.877069   0.617249
 0.933193   0.968544
 0.0534829  0.115737
 0.729433   0.14046
 0.19727    0.947731
 0.146534   0.0746781
 0.516934   0.57386

[:, :, 3] =
 0.832656  0.993766
 0.245658  0.564889
 0.881817  0.325685
 0.182307  0.782469
 0.884893  0.909196
 0.517765  0.587938
 0.39805   0.225796
 0.151747  0.785742
 0.817745  0.886162
 0.321821  0.469467

 julia> chain = Chains(val, [:s, :m]) # construct a chain of samples using MCMCChains
Chains MCMC chain (10×2×3 Array{Float64, 3}):

Iterations        = 1:1:10
Number of chains  = 3
Samples per chain = 10
parameters        = s, m

Summary Statistics
  parameters      mean       std   naive_se      mcse          ess      rhat 
      Symbol   Float64   Float64    Float64   Float64      Float64   Float64 

           s    0.5122    0.2884     0.0527    0.0477   -1435.6207    0.9789
           m    0.5924    0.2965     0.0541    0.0680      29.9515    1.0764

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           s    0.0954    0.2094    0.5242    0.7957    0.8982
           m    0.0722    0.3530    0.6572    0.7849    0.9950

julia> logprior(demo_model(x), chain)
10×3 Matrix{Float64}:
    -2.65353   -4.39497   -2.2767
    -1.78603   -8.57528   -6.66996
    -5.27324   -2.03405   -1.74373
    -1.89871   -1.9003   -10.8995
    -1.80472   -2.1971    -2.15103
    -2.27175  -44.6902    -2.54584
    -2.56001   -1.74381   -3.09837
-10.4215   -10.5247   -13.9263
    -2.04761  -12.492     -2.16627
-20.5141    -2.53425   -4.41793
```   
"""
function logprior(
    model_instance::Model,
    chain::AbstractMCMC.AbstractChains,
    start_idx::Int = 1,
)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(
        Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3)),
    ) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end


"""
    loglikelihoods(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log likelihoods evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> @model function demo_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
    end
end
demo_model (generic function with 2 methods)

julia> using StableRNGs, MCMCChains

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3)
10×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.58566    0.154556
 0.686969   0.847473
 0.863187   0.445798
 0.472695   0.556557
 0.887645   0.481231
 0.209934   0.248418
 0.478046   0.435529
 0.426059   0.804087
 0.913847   0.944763
 0.0162749  0.414059

[:, :, 2] =
 0.222991   0.929401
 0.514089   0.766995
 0.979532   0.171396
 0.864463   0.617206
 0.924191   0.371671
 0.907488   0.534833
 0.578085   0.288955
 0.431816   0.199648
 0.0709966  0.348349
 0.324199   0.931152

[:, :, 3] =
 0.906302  0.39177
 0.777222  0.578203
 0.613677  0.980324
 0.520658  0.885295
 0.112626  0.0805126
 0.117799  0.596072
 0.505948  0.308587
 0.893709  0.124035
 0.225277  0.743494
 0.377099  0.317035

 julia> chain = Chains(val, [:s, :m])
Chains MCMC chain (10×2×3 Array{Float64, 3}):

Iterations        = 1:1:10
Number of chains  = 3
Samples per chain = 10
parameters        = s, m

Summary Statistics
  parameters      mean       std   naive_se      mcse       ess      rhat 
      Symbol   Float64   Float64    Float64   Float64   Float64   Float64 

           s    0.5469    0.2973     0.0543    0.0547   30.4571    1.0020
           m    0.5166    0.2751     0.0502    0.0280   86.4365    0.9243

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           s    0.0559    0.3374    0.5174    0.8641    0.9394
           m    0.1121    0.3107    0.4635    0.7611    0.9545

julia> loglikelihoods(demo_model(x), chain)
10×3 Matrix{Float64}:
    -2053.82  -2289.0   -1587.9
    -1431.8   -1551.34  -1508.8
    -1563.54  -1732.84  -1442.21
    -1740.78  -1471.96  -1507.89
    -1536.64  -1597.04  -7694.37
    -3697.44  -1504.31  -4511.38
    -1858.51  -1886.68  -1970.19
    -1638.88  -2316.02  -1810.87
    -1390.37  -9144.19  -2400.74
-38045.7   -1813.85  -2283.61
```  
"""
function loglikelihoods(
    model_instance::Model,
    chain::AbstractMCMC.AbstractChains,
    start_idx::Int = 1,
)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(
        Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3)),
    ) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict)
    end
end


"""
    logjoint(model_instance::Model, chain::AbstractMCMC.AbstractChains, start_idx::Int)

Return an array of log posteriors evaluated at each sample in an MCMC chain or sample array.

Example:
    
```jldoctest
julia> using StableRNGs, MCMCChains

julia> rng = StableRNG(123)
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> val = rand(rng, 10, 2, 3)
10×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.775123   0.888009
 0.0348401  0.328242
 0.0419726  0.0832775
 0.434804   0.828488
 0.990519   0.791443
 0.303966   0.559475
 0.256094   0.145148
 0.0708956  0.609441
 0.9972     0.209455
 0.215559   0.755402

[:, :, 2] =
 0.109065   0.33411
 0.907512   0.715396
 0.0931766  0.583011
 0.166632   0.0419226
 0.344997   0.308126
 0.771344   0.322862
 0.90522    0.743365
 0.692612   0.819923
 0.650188   0.107918
 0.458714   0.742119

[:, :, 3] =
 0.398571  0.0512853
 0.384821  0.408064
 0.295453  0.0493099
 0.299926  0.141914
 0.447855  0.979068
 0.490272  0.918014
 0.739946  0.381837
 0.780228  0.25193
 0.595156  0.96152
 0.22909   0.626672

 julia> chain = Chains(val, [:s, :m])
 Chains MCMC chain (10×2×3 Array{Float64, 3}):
 
 Iterations        = 1:1:10
 Number of chains  = 3
 Samples per chain = 10
 parameters        = s, m
 
 Summary Statistics
   parameters      mean       std   naive_se      mcse       ess      rhat 
       Symbol   Float64   Float64    Float64   Float64   Float64   Float64 
 
            s    0.4627    0.2972     0.0543    0.0585   42.0294    1.0201
            m    0.4896    0.3117     0.0569    0.0476   95.8121    0.9652
 
 Quantiles
   parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
       Symbol   Float64   Float64   Float64   Float64   Float64 
 
            s    0.0400    0.2358    0.4167    0.7281    0.9924
            m    0.0473    0.2201    0.4838    0.7524    0.9663

julia> logjoint(demo_model(x), chain, 2)
9×3 Matrix{Float64}:
    -19148.6   -1433.33  -2111.14
    -20443.2   -5706.89  -3410.79
    -1619.63  -5566.35  -3095.36
    -1411.85  -2432.85  -1572.57
    -2184.95  -1691.38  -1531.74
    -3481.6   -1425.4   -1657.91
    -7302.75  -1438.73  -1750.86
    -1698.33  -2030.47  -1454.15
    -2474.31  -1626.86  -2535.97
```   
"""
function logjoint(
    model_instance::Model,
    chain::AbstractMCMC.AbstractChains,
    start_idx::Int = 1,
)
    vi = VarInfo(model_instance) # extract variables info from the model
    map(
        Iterators.product(start_idx:size(chain, 1), 1:size(chain, 3)),
    ) do (iteration_idx, chain_idx)
        argvals_dict = OrderedDict(
            vn => chain[iteration_idx, Symbol(vn), chain_idx] for vn_parent in keys(vi)
            for vn in DynamicPPL.TestUtils.varname_leaves(vn_parent, vi[vn_parent])
        )
        Distributions.loglikelihood(model_instance, argvals_dict) +
        DynamicPPL.logprior(model_instance, argvals_dict)
    end
end
