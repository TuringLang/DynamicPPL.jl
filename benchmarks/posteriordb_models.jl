# Performance notes for future changes:
# - Preserve semantic structure: homogeneous `Fill`/`IsoNormal` products let
#   Bijectors and Mooncake use one transform or adjoint instead of one per datum.
# - If an algorithm needs a native rule, attach it to a public distribution or
#   solver boundary, not a private likelihood helper or incidental array operation.
# - Track reverse-tape allocation first; it predicts gradient time better than
#   parameter count or primal time in this catalog.
# - Concretize and prepare data in `make_model`; `_FixedData` expresses a
#   semantically fixed AD boundary (notably solver times) and avoids material
#   data cotangents where they matter.
# - Match representation to the AD boundary: keep homogeneous products lazy,
#   but materialize the few active matrix-valued repeated distributions.
# - Prefer public Stats/Distributions batch boundaries such as `logsumexp`,
#   `product_distribution`, and `loglikelihood(MvNormal(...), X)`.
# - Keep catalog rules few and together at the end of this file: only the
#   Kronecker distribution boundary and fixed-data activity marker live here.

import Bijectors
using Bijectors: ordered
using Distributions
using FillArrays: Fill
using LinearAlgebra
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqLowOrderRK: DP5
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve
import SciMLSensitivity
using SparseArrays
using StatsFuns: logaddexp, logistic, logit, log1pexp, logsumexp, softmax,
    loggamma, xlogy
using Statistics
using DynamicPPL

struct _FixedData{T}
    value::T
end


# DynamicPPL's improper priors have simpler supports than Bijectors' generic
# continuous-distribution fallback can express. These specialisations avoid an
# unnecessary run-time boundedness branch and expose homogeneous product maps.
Bijectors.VectorBijectors.scalar_to_scalar_bijector(::Flat) =
    Bijectors.VectorBijectors.TypedIdentity()
Bijectors.VectorBijectors.scalar_to_scalar_bijector(d::FlatPos) =
    Bijectors.VectorBijectors.Log(d.l, 1)

const POSTERIOR_NAMES = sort!([
    "arma-arma11",
    "arK-arK",
    "bball_drive_event_0-hmm_drive_0",
    "bball_drive_event_1-hmm_drive_1",
    "diamonds-diamonds",
    "earnings-earn_height",
    "earnings-log10earn_height",
    "earnings-logearn_height",
    "earnings-logearn_height_male",
    "earnings-logearn_interaction",
    "earnings-logearn_interaction_z",
    "earnings-logearn_logheight_male",
    "eight_schools-eight_schools_centered",
    "eight_schools-eight_schools_noncentered",
    "garch-garch11",
    "gp_pois_regr-gp_pois_regr",
    "gp_pois_regr-gp_regr",
    "hmm_example-hmm_example",
    "hudson_lynx_hare-lotka_volterra",
    "kidiq-kidscore_interaction",
    "kidiq-kidscore_momhs",
    "kidiq-kidscore_momhsiq",
    "kidiq-kidscore_momiq",
    "kidiq_with_mom_work-kidscore_interaction_c",
    "kidiq_with_mom_work-kidscore_interaction_c2",
    "kidiq_with_mom_work-kidscore_interaction_z",
    "kidiq_with_mom_work-kidscore_mom_work",
    "kilpisjarvi_mod-kilpisjarvi",
    "low_dim_gauss_mix-low_dim_gauss_mix",
    "mcycle_gp-accel_gp",
    "mesquite-logmesquite",
    "mesquite-logmesquite_logva",
    "mesquite-logmesquite_logvas",
    "mesquite-logmesquite_logvash",
    "mesquite-logmesquite_logvolume",
    "mesquite-mesquite",
    "nes1972-nes",
    "nes1976-nes",
    "nes1980-nes",
    "nes1984-nes",
    "nes1988-nes",
    "nes1992-nes",
    "nes1996-nes",
    "nes2000-nes",
    "one_comp_mm_elim_abs-one_comp_mm_elim_abs",
    "sblrc-blr",
    "sblri-blr",
    "GLMM_Poisson_data-GLMM_Poisson_model",
    "GLMM_data-GLMM1_model",
    "GLM_Binomial_data-GLM_Binomial_model",
    "GLM_Poisson_Data-GLM_Poisson_model",
    "M0_data-M0_model",
    "Mb_data-Mb_model",
    "Mh_data-Mh_model",
    "Mt_data-Mt_model",
    "Mtbh_data-Mtbh_model",
    "Mth_data-Mth_model",
    "Rate_1_data-Rate_1_model",
    "Rate_2_data-Rate_2_model",
    "Rate_3_data-Rate_3_model",
    "Rate_4_data-Rate_4_model",
    "Rate_5_data-Rate_5_model",
    "Survey_data-Survey_model",
    "bones_data-bones_model",
    "butterfly-multi_occupancy",
    "dogs-dogs",
    "dogs-dogs_hierarchical",
    "dogs-dogs_log",
    "dogs-dogs_nonhierarchical",
    "dugongs_data-dugongs_model",
    "ecdc0401-covid19imperial_v2",
    "ecdc0401-covid19imperial_v3",
    "ecdc0501-covid19imperial_v2",
    "ecdc0501-covid19imperial_v3",
    "election88-election88_full",
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
    "hmm_gaussian_simulated-hmm_gaussian",
    "iohmm_reg_simulated-iohmm_reg",
    "irt_2pl-irt_2pl",
    "loss_curves-losscurve_sislob",
    "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse",
    "lsat_data-lsat_model",
    "mcycle_splines-accel_splines",
    "mnist-nn_rbm1bJ100",
    "mnist_100-nn_rbm1bJ10",
    "nes_logit_data-nes_logit_model",
    "normal_2-normal_mixture",
    "normal_5-normal_mixture_k",
    "ovarian-logistic_regression_rhs",
    "pilots-pilots",
    "prideprejudice_chapter-ldaK5",
    "prideprejudice_paragraph-ldaK5",
    "prostate-logistic_regression_rhs",
    "radon_all-radon_county_intercept",
    "radon_all-radon_hierarchical_intercept_centered",
    "radon_all-radon_hierarchical_intercept_noncentered",
    "radon_all-radon_partially_pooled_centered",
    "radon_all-radon_partially_pooled_noncentered",
    "radon_all-radon_pooled",
    "radon_all-radon_variable_intercept_centered",
    "radon_all-radon_variable_intercept_noncentered",
    "radon_all-radon_variable_intercept_slope_centered",
    "radon_all-radon_variable_intercept_slope_noncentered",
    "radon_all-radon_variable_slope_centered",
    "radon_all-radon_variable_slope_noncentered",
    "radon_mn-radon_county_intercept",
    "radon_mn-radon_hierarchical_intercept_centered",
    "radon_mn-radon_hierarchical_intercept_noncentered",
    "radon_mn-radon_partially_pooled_centered",
    "radon_mn-radon_partially_pooled_noncentered",
    "radon_mn-radon_pooled",
    "radon_mn-radon_variable_intercept_centered",
    "radon_mn-radon_variable_intercept_noncentered",
    "radon_mn-radon_variable_intercept_slope_centered",
    "radon_mn-radon_variable_intercept_slope_noncentered",
    "radon_mn-radon_variable_slope_centered",
    "radon_mn-radon_variable_slope_noncentered",
    "radon_mod-radon_county",
    "rats_data-rats_model",
    "rstan_downloads-prophet",
    "sat-hier_2pl",
    "science_irt-grsm_latent_reg_irt",
    "seeds_data-seeds_centered_model",
    "seeds_data-seeds_model",
    "seeds_data-seeds_stanified_model",
    "sesame_data-sesame_one_pred_a",
    "sir-sir",
    "soil_carbon-soil_incubation",
    "state_wide_presidential_votes-hierarchical_gp",
    "surgical_data-surgical_model",
    "synthetic_grid_RBF_kernels-kronecker_gp",
    "three_docs1200-ldaK2",
    "three_men1-ldaK2",
    "three_men2-ldaK2",
    "three_men3-ldaK2",
    "timssAusTwn_irt-gpcm_latent_reg_irt",
    "traffic_accident_nyc-bym2_offset_only",
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
    "wells_data-wells_daae_c_model",
    "wells_data-wells_dae_c_model",
    "wells_data-wells_dae_inter_model",
    "wells_data-wells_dae_model",
    "wells_data-wells_dist",
    "wells_data-wells_dist100_model",
    "wells_data-wells_dist100ars_model",
    "wells_data-wells_interaction_c_model",
    "wells_data-wells_interaction_model",
])

"""
    RBFKroneckerNormal(variance, bandwidth, row_cholesky, nugget, distances)

Zero-mean matrix distribution with covariance `Krbf ⊗ Krow + nugget * I`, where
`Krbf = variance * exp.(distances * bandwidth) + 1e-5I` and
`Krow = row_cholesky * row_cholesky'`. The additive nugget means this is not a
`MatrixNormal`, whose covariance must be exactly separable.
"""
struct RBFKroneckerNormal{T<:Real,TL<:AbstractMatrix,TD<:AbstractMatrix} <:
       ContinuousMatrixDistribution
    variance::T
    bandwidth::T
    row_cholesky::TL
    nugget::T
    distances::TD
end

Base.size(d::RBFKroneckerNormal) =
    (size(d.row_cholesky, 1), size(d.distances, 1))
Distributions.insupport(d::RBFKroneckerNormal, y::AbstractMatrix) = size(y) == size(d)
Distributions.params(d::RBFKroneckerNormal) = (
    d.variance, d.bandwidth, d.row_cholesky, d.nugget, d.distances,
)

function Distributions._logpdf(d::RBFKroneckerNormal, y::AbstractMatrix{<:Real})
    column_cov = Symmetric(
        d.variance .* exp.(d.distances .* d.bandwidth) + 1.0e-5I,
    )
    row_cov = Symmetric(d.row_cholesky * d.row_cholesky')
    row_values, row_vectors = eigen(row_cov)
    column_values, column_vectors = eigen(column_cov)
    variances = row_values .* column_values' .+ d.nugget
    rotated = row_vectors' * y * column_vectors
    return -0.5 * (
        length(y) * log(2pi) +
        sum(abs2.(rotated) ./ variances) +
        sum(log, variances)
    )
end

function gp_exp_quad_cov(x, sigma, length_scale)
    return [sigma^2 * exp(-0.5 * ((a - b) / length_scale)^2) for a in x, b in x]
end

function _solve_ode(
    algorithm, f, y0, t0, ts, parameters, abstol, reltol, maxiters,
)
    save_times = ts.value
    problem = ODEProblem(f, y0, (t0.value, save_times[end]), parameters)
    solution = solve(
        problem, algorithm;
        saveat=save_times, abstol, reltol, maxiters,
        sensealg=SciMLSensitivity.ForwardDiffSensitivity(),
    )
    return reduce(hcat, solution.u)'
end

@model function arma_arma11(T, y)
    mu ~ Normal(0, 10)
    phi ~ Normal(0, 2)
    theta ~ Normal(0, 2)
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)

    nu = Vector{typeof(mu)}(undef, T)
    err = Vector{typeof(mu)}(undef, T)
    nu[1] = mu + phi * mu
    err[1] = y[1] - nu[1]
    for t in 2:T
        nu[t] = mu + phi * y[t - 1] + theta * err[t - 1]
        err[t] = y[t] - nu[t]
    end
    y ~ product_distribution(map(Normal, nu, Fill(sigma, T)))
end

function make_model(::Val{Symbol("arma-arma11")}, data)
    return arma_arma11(data["T"], Float64.(data["y"]))
end


@model function arK_arK(K, T, y)
    alpha ~ Normal(0, 10)
    beta ~ product_distribution(Fill(Normal(0, 10), K))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    for t in (K + 1):T
        mu = alpha
        for k in 1:K
            mu += beta[k] * y[t - k]
        end
        y[t] ~ Normal(mu, sigma)
    end
end

function make_model(::Val{Symbol("arK-arK")}, data)
    return arK_arK(data["K"], data["T"], Float64.(data["y"]))
end


@model function earnings_earn_height(earn, height)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* height
    earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-earn_height")}, data)
    return earnings_earn_height(Float64.(data["earn"]), Float64.(data["height"]))
end


@model function earnings_log10earn_height(log10_earn, height)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* height
    log10_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-log10earn_height")}, data)
    return earnings_log10earn_height(log10.(Float64.(data["earn"])), Float64.(data["height"]))
end


@model function earnings_logearn_height(log_earn, height)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* height
    log_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-logearn_height")}, data)
    return earnings_logearn_height(log.(Float64.(data["earn"])), Float64.(data["height"]))
end


@model function earnings_logearn_height_male(log_earn, height, male)
    beta ~ product_distribution(Fill(Flat(), 3))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* height .+ beta[3] .* male
    log_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-logearn_height_male")}, data)
    return earnings_logearn_height_male(
        log.(Float64.(data["earn"])), Float64.(data["height"]), Float64.(data["male"]),
    )
end


@model function earnings_logearn_interaction(log_earn, height, male, interaction)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* height .+ beta[3] .* male .+ beta[4] .* interaction
    log_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-logearn_interaction")}, data)
    height = Float64.(data["height"])
    male = Float64.(data["male"])
    return earnings_logearn_interaction(
        log.(Float64.(data["earn"])), height, male, height .* male,
    )
end


@model function earnings_logearn_interaction_z(log_earn, z_height, male, interaction)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* z_height .+ beta[3] .* male .+ beta[4] .* interaction
    log_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-logearn_interaction_z")}, data)
    height = Float64.(data["height"])
    male = Float64.(data["male"])
    z_height = (height .- mean(height)) ./ std(height)
    return earnings_logearn_interaction_z(
        log.(Float64.(data["earn"])), z_height, male, z_height .* male,
    )
end


@model function earnings_logearn_logheight_male(log_earn, log_height, male)
    beta ~ product_distribution(Fill(Flat(), 3))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* log_height .+ beta[3] .* male
    log_earn ~ MvNormal(mu, sigma)
end

function make_model(::Val{Symbol("earnings-logearn_logheight_male")}, data)
    return earnings_logearn_logheight_male(
        log.(Float64.(data["earn"])), log.(Float64.(data["height"])), Float64.(data["male"]),
    )
end


@model function eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta ~ product_distribution(Fill(Normal(mu, tau), J))
    for i in 1:J
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

function make_model(::Val{Symbol("eight_schools-eight_schools_centered")}, data)
    return eight_schools_centered(data["J"], Float64.(data["y"]), Float64.(data["sigma"]))
end


@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta_trans ~ product_distribution(Fill(Normal(), J))
    for i in 1:J
        y[i] ~ Normal(theta_trans[i] * tau + mu, sigma[i])
    end
end

function make_model(::Val{Symbol("eight_schools-eight_schools_noncentered")}, data)
    return eight_schools_noncentered(data["J"], Float64.(data["y"]), Float64.(data["sigma"]))
end


@model function garch_garch11(T, y, sigma1)
    mu ~ Flat()
    alpha0 ~ FlatPos(0.0)
    alpha1 ~ Uniform(0, 1)
    beta1 ~ Flat()
    if !(0 <= beta1 <= 1 - alpha1)
        DynamicPPL.@addlogprob! -Inf
        return
    end

    sigma = Vector{typeof(mu)}(undef, T)
    sigma[1] = sigma1
    for t in 2:T
        sigma[t] = sqrt(alpha0 + alpha1 * (y[t - 1] - mu)^2 + beta1 * sigma[t - 1]^2)
    end
    for t in 1:T
        y[t] ~ Normal(mu, sigma[t])
    end
end

function make_model(::Val{Symbol("garch-garch11")}, data)
    return garch_garch11(data["T"], Float64.(data["y"]), Float64(data["sigma1"]))
end


@model function gp_pois_regr(dist_sq, k, N)
    dist_sq = dist_sq.value
    rho ~ Gamma(25, 1 / 4)
    alpha ~ truncated(Normal(0, 2); lower=0)
    f_tilde ~ product_distribution(Fill(Normal(), N))
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + 1e-10 * I
    factor = cholesky(Symmetric(covariance); check=false)
    if !issuccess(factor)
        DynamicPPL.@addlogprob! -Inf
        return
    end
    f = factor.L * f_tilde
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), f)), k,
    )
end

function make_model(::Val{Symbol("gp_pois_regr-gp_pois_regr")}, data)
    x = Float64.(data["x"])
    return gp_pois_regr(_FixedData((x .- x') .^ 2), Int.(data["k"]), data["N"])
end


@model function gp_regr(dist_sq, y, N)
    dist_sq = dist_sq.value
    y = y.value
    rho ~ Gamma(25, 1 / 4)
    alpha ~ truncated(Normal(0, 2); lower=0)
    sigma ~ truncated(Normal(0, 1); lower=0)
    covariance = alpha^2 .* exp.(-0.5 .* dist_sq ./ rho^2) + sigma * I
    y ~ MvNormal(zeros(N), Symmetric(covariance))
end

function make_model(::Val{Symbol("gp_pois_regr-gp_regr")}, data)
    x = Float64.(data["x"])
    return gp_regr(
        _FixedData((x .- x') .^ 2), _FixedData(Float64.(data["y"])), data["N"],
    )
end


@model function kidiq_kidscore_momhs(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq-kidscore_momhs")}, data)
    mom_hs = Float64.(data["mom_hs"])
    return kidiq_kidscore_momhs(
        Float64.(data["kid_score"]), hcat(ones(length(mom_hs)), mom_hs),
    )
end


@model function kidiq_kidscore_momiq(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq-kidscore_momiq")}, data)
    mom_iq = Float64.(data["mom_iq"])
    return kidiq_kidscore_momiq(
        Float64.(data["kid_score"]), hcat(ones(length(mom_iq)), mom_iq),
    )
end


@model function kidiq_kidscore_momhsiq(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 3))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq-kidscore_momhsiq")}, data)
    mom_hs = Float64.(data["mom_hs"])
    mom_iq = Float64.(data["mom_iq"])
    return kidiq_kidscore_momhsiq(
        Float64.(data["kid_score"]), hcat(ones(length(mom_hs)), mom_hs, mom_iq),
    )
end


@model function kidiq_kidscore_interaction(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ truncated(Cauchy(0, 2.5); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq-kidscore_interaction")}, data)
    mom_hs = Float64.(data["mom_hs"])
    mom_iq = Float64.(data["mom_iq"])
    X = hcat(ones(length(mom_hs)), mom_hs, mom_iq, mom_hs .* mom_iq)
    return kidiq_kidscore_interaction(Float64.(data["kid_score"]), X)
end


@model function kidiq_kidscore_interaction_c(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq_with_mom_work-kidscore_interaction_c")}, data)
    mom_hs = Float64.(data["mom_hs"])
    mom_iq = Float64.(data["mom_iq"])
    centered_hs = mom_hs .- mean(mom_hs)
    centered_iq = mom_iq .- mean(mom_iq)
    X = hcat(
        ones(length(mom_hs)), centered_hs, centered_iq, centered_hs .* centered_iq,
    )
    return kidiq_kidscore_interaction_c(Float64.(data["kid_score"]), X)
end


@model function kidiq_kidscore_interaction_c2(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq_with_mom_work-kidscore_interaction_c2")}, data)
    centered_hs = Float64.(data["mom_hs"]) .- 0.5
    centered_iq = Float64.(data["mom_iq"]) .- 100
    X = hcat(
        ones(length(centered_hs)), centered_hs, centered_iq,
        centered_hs .* centered_iq,
    )
    return kidiq_kidscore_interaction_c2(Float64.(data["kid_score"]), X)
end


@model function kidiq_kidscore_interaction_z(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq_with_mom_work-kidscore_interaction_z")}, data)
    mom_hs = Float64.(data["mom_hs"])
    mom_iq = Float64.(data["mom_iq"])
    z_hs = (mom_hs .- mean(mom_hs)) ./ (2std(mom_hs))
    z_iq = (mom_iq .- mean(mom_iq)) ./ (2std(mom_iq))
    X = hcat(ones(length(z_hs)), z_hs, z_iq, z_hs .* z_iq)
    return kidiq_kidscore_interaction_z(Float64.(data["kid_score"]), X)
end


@model function kidiq_kidscore_mom_work(kid_score, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(kid_score))),
        kid_score .- X * beta,
    )
end

function make_model(::Val{Symbol("kidiq_with_mom_work-kidscore_mom_work")}, data)
    mom_work = Int.(data["mom_work"])
    X = hcat(
        ones(length(mom_work)), mom_work .== 2, mom_work .== 3, mom_work .== 4,
    )
    return kidiq_kidscore_mom_work(Float64.(data["kid_score"]), Float64.(X))
end


@model function kilpisjarvi(x, y, pmualpha, psalpha, pmubeta, psbeta)
    alpha ~ Normal(pmualpha, psalpha)
    beta ~ Normal(pmubeta, psbeta)
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(y))),
        y .- (alpha .+ beta .* x),
    )
end

function make_model(::Val{Symbol("kilpisjarvi_mod-kilpisjarvi")}, data)
    return kilpisjarvi(
        Float64.(data["x"]), Float64.(data["y"]), Float64(data["pmualpha"]),
        Float64(data["psalpha"]), Float64(data["pmubeta"]), Float64(data["psbeta"]),
    )
end


@model function mesquite_logmesquite(log_weight, X)
    beta ~ product_distribution(Fill(Flat(), 7))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(log_weight))),
        log_weight .- X * beta,
    )
end

function make_model(::Val{Symbol("mesquite-logmesquite")}, data)
    X = hcat(
        ones(length(data["weight"])), log.(data["diam1"]), log.(data["diam2"]),
        log.(data["canopy_height"]), log.(data["total_height"]),
        log.(data["density"]), data["group"],
    )
    return mesquite_logmesquite(log.(Float64.(data["weight"])), Float64.(X))
end


@model function mesquite_logmesquite_logva(log_weight, X)
    beta ~ product_distribution(Fill(Flat(), 4))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(log_weight))),
        log_weight .- X * beta,
    )
end

function make_model(::Val{Symbol("mesquite-logmesquite_logva")}, data)
    log_volume = log.(data["diam1"] .* data["diam2"] .* data["canopy_height"])
    log_area = log.(data["diam1"] .* data["diam2"])
    X = hcat(ones(length(log_volume)), log_volume, log_area, data["group"])
    return mesquite_logmesquite_logva(log.(Float64.(data["weight"])), Float64.(X))
end


@model function mesquite_logmesquite_logvas(log_weight, X)
    beta ~ product_distribution(Fill(Flat(), 7))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(log_weight))),
        log_weight .- X * beta,
    )
end

function make_model(::Val{Symbol("mesquite-logmesquite_logvas")}, data)
    log_volume = log.(data["diam1"] .* data["diam2"] .* data["canopy_height"])
    log_area = log.(data["diam1"] .* data["diam2"])
    X = hcat(
        ones(length(log_volume)), log_volume, log_area,
        log.(data["diam1"] ./ data["diam2"]), log.(data["total_height"]),
        log.(data["density"]), data["group"],
    )
    return mesquite_logmesquite_logvas(log.(Float64.(data["weight"])), Float64.(X))
end


@model function mesquite_logmesquite_logvash(log_weight, X)
    beta ~ product_distribution(Fill(Flat(), 6))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(log_weight))),
        log_weight .- X * beta,
    )
end

function make_model(::Val{Symbol("mesquite-logmesquite_logvash")}, data)
    log_volume = log.(data["diam1"] .* data["diam2"] .* data["canopy_height"])
    log_area = log.(data["diam1"] .* data["diam2"])
    X = hcat(
        ones(length(log_volume)), log_volume, log_area,
        log.(data["diam1"] ./ data["diam2"]), log.(data["total_height"]), data["group"],
    )
    return mesquite_logmesquite_logvash(log.(Float64.(data["weight"])), Float64.(X))
end


@model function mesquite_logmesquite_logvolume(log_weight, X)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(log_weight))),
        log_weight .- X * beta,
    )
end

function make_model(::Val{Symbol("mesquite-logmesquite_logvolume")}, data)
    log_volume = log.(data["diam1"] .* data["diam2"] .* data["canopy_height"])
    X = hcat(ones(length(log_volume)), log_volume)
    return mesquite_logmesquite_logvolume(
        log.(Float64.(data["weight"])), Float64.(X),
    )
end


@model function mesquite_mesquite(
    weight, diam1, diam2, canopy_height, total_height, density, group
)
    beta ~ product_distribution(Fill(Flat(), 7))
    sigma ~ FlatPos(0.0)
    mu = beta[1] .+ beta[2] .* diam1 .+ beta[3] .* diam2 .+
         beta[4] .* canopy_height .+ beta[5] .* total_height .+
         beta[6] .* density .+ beta[7] .* group
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(weight))), weight .- mu,
    )
end

function make_model(::Val{Symbol("mesquite-mesquite")}, data)
    return mesquite_mesquite(
        Float64.(data["weight"]), Float64.(data["diam1"]), Float64.(data["diam2"]),
        Float64.(data["canopy_height"]), Float64.(data["total_height"]),
        Float64.(data["density"]), Float64.(data["group"]),
    )
end


@model function nes_model(partyid7, real_ideo, race_adj, educ1, gender, income, age_discrete)
    beta ~ product_distribution(Fill(Flat(), 9))
    sigma ~ FlatPos(0.0)
    age30_44 = age_discrete .== 2
    age45_64 = age_discrete .== 3
    age65up = age_discrete .== 4
    mu = beta[1] .+ beta[2] .* real_ideo .+ beta[3] .* race_adj .+
         beta[4] .* age30_44 .+ beta[5] .* age45_64 .+ beta[6] .* age65up .+
         beta[7] .* educ1 .+ beta[8] .* gender .+ beta[9] .* income
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(partyid7))), partyid7 .- mu,
    )
end

function _make_nes(data)
    return nes_model(
        Float64.(data["partyid7"]), Float64.(data["real_ideo"]), Float64.(data["race_adj"]),
        Float64.(data["educ1"]), Float64.(data["gender"]), Float64.(data["income"]),
        Int.(data["age_discrete"]),
    )
end

make_model(::Val{Symbol("nes1972-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1976-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1980-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1984-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1988-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1992-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes1996-nes")}, data) = _make_nes(data)

make_model(::Val{Symbol("nes2000-nes")}, data) = _make_nes(data)


@model function sblrc_blr(X, y, D)
    X = X.value
    y = y.value
    beta ~ product_distribution(Fill(Normal(0, 10), D))
    sigma ~ truncated(Normal(0, 10); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(y))), y .- X * beta,
    )
end

function make_model(::Val{Symbol("sblrc-blr")}, data)
    return sblrc_blr(
        _FixedData(Float64.(data["X"])), _FixedData(Float64.(data["y"])), data["D"],
    )
end


@model function sblri_blr(X, y, D)
    X = X.value
    y = y.value
    beta ~ product_distribution(Fill(Normal(0, 10), D))
    sigma ~ truncated(Normal(0, 10); lower=0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(y))), y .- X * beta,
    )
end

function make_model(::Val{Symbol("sblri-blr")}, data)
    return sblri_blr(
        _FixedData(Float64.(data["X"])), _FixedData(Float64.(data["y"])), data["D"],
    )
end


@model function diamonds_model(stats, prior_only)
    data = stats.value
    b ~ product_distribution(Fill(Normal(), size(data.XtX, 1)))
    Intercept ~ LocationScale(8, 10, TDist(3))
    sigma ~ truncated(LocationScale(0, 10, TDist(3)); lower=0)
    if prior_only == 0
        sse = data.yty - 2dot(b, data.Xty) - 2Intercept * data.ysum +
            dot(b, data.XtX * b) + 2Intercept * dot(data.xsum, b) +
            data.n * Intercept^2
        DynamicPPL.@addlogprob! -0.5data.n * log(2pi) - data.n * log(sigma) -
            0.5sse / sigma^2
    end
end

function make_model(::Val{Symbol("diamonds-diamonds")}, data)
    X = Float64.(data["X"])
    centered_X = X[:, 2:end] .- mean(X[:, 2:end]; dims=1)
    y = Float64.(data["Y"])
    stats = _FixedData((
        XtX=centered_X' * centered_X,
        Xty=centered_X' * y,
        xsum=vec(sum(centered_X; dims=1)),
        ysum=sum(y), yty=sum(abs2, y), n=length(y),
    ))
    return diamonds_model(stats, data["prior_only"])
end


@model function low_dim_gauss_mix(y, N)
    mu ~ ordered(product_distribution(Fill(Flat(), 2)))
    sigma ~ product_distribution(Fill(truncated(Normal(0, 2); lower=0), 2))
    theta ~ Beta(5, 5)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 2), mu[1]) + logpdf(Normal(0, 2), mu[2])
    for n in 1:N
        component1 = log(theta) + logpdf(Normal(mu[1], sigma[1]), y[n])
        component2 = log1p(-theta) + logpdf(Normal(mu[2], sigma[2]), y[n])
        DynamicPPL.@addlogprob! logaddexp(component1, component2)
    end
end

function make_model(::Val{Symbol("low_dim_gauss_mix-low_dim_gauss_mix")}, data)
    return low_dim_gauss_mix(Float64.(data["y"]), data["N"])
end


"""
    HiddenMarkovModel(
        n, initial, transition, emissions, transition_layout,
        first_observation=Val(:ordinary),
    )

Joint distribution of an observation sequence after marginalising its discrete
Markov states. `emissions` is either one ordinary Distributions.jl distribution
per state, or an `n × K` matrix for time-varying state emissions. Multivariate
state emissions should use `product_distribution`; observations across time are
not independent.

The `:summed_first` first-observation option and `:source_logprob` transition
layout reproduce two nonstandard PosteriorDB Stan likelihoods exactly.
"""
struct HiddenMarkovModel{I,A,E,M,F} <: ContinuousMatrixDistribution
    n::Int
    initial::I
    transition::A
    emissions::E
    transition_layout::Val{M}
    first_observation::Val{F}
end

struct NormalEmissions{M,S}
    means::M
    scales::S
end

HiddenMarkovModel(n, initial, transition, emissions, transition_layout) =
    HiddenMarkovModel(
        n, initial, transition, emissions, transition_layout, Val(:ordinary),
    )

function Base.size(d::HiddenMarkovModel)
    dimension = if d.emissions isa NormalEmissions
        1
    else
        emission = first(d.emissions)
        emission isa UnivariateDistribution ? 1 : length(emission)
    end
    return (d.n, dimension)
end
Distributions.insupport(d::HiddenMarkovModel, y::AbstractMatrix) =
    size(y) == size(d)
Distributions.params(d::HiddenMarkovModel) = (
    d.initial, d.transition, d.emissions, d.transition_layout,
    d.first_observation,
)

function Distributions._logpdf(
    d::HiddenMarkovModel,
    observations::AbstractMatrix{<:Real},
)
    size(observations) == size(d) || return -Inf
    T, K = d.n, length(d.initial)
    emission = if d.emissions isa NormalEmissions
        means = d.emissions.means isa AbstractVector ?
            reshape(d.emissions.means, 1, K) : d.emissions.means
        scales = reshape(d.emissions.scales, 1, K)
        residual = (observations[:, 1] .- means) ./ scales
        -0.5log(2pi) .- log.(scales) .- 0.5 .* abs2.(residual)
    else
        values = Matrix{Float64}(undef, T, K)
        for k in 1:K, t in 1:T
            distribution = d.emissions isa AbstractVector ?
                d.emissions[k] : d.emissions[t, k]
            observation = distribution isa UnivariateDistribution ?
                observations[t, 1] : view(observations, t, :)
            values[t, k] = logpdf(distribution, observation)
        end
        values
    end
    if d.first_observation isa Val{:summed_first}
        initial_emission = sum(view(emission, 1, :))
        emission[1, :] .= initial_emission
    end
    offset = maximum(emission[1, :])
    alpha = d.initial .* exp.(emission[1, :] .- offset)
    scale = sum(alpha)
    value = offset + log(scale)
    alpha = alpha ./ scale
    transition = d.transition_layout isa Val{:rows} ?
        permutedims(d.transition) : d.transition
    for t in 2:T
        offset = maximum(emission[t, :])
        emission_probability = exp.(emission[t, :] .- offset)
        predicted = if d.transition_layout isa Val{:source_logprob}
            sum(alpha .* exp.(transition[t, :]))
        else
            transition * alpha
        end
        alpha = predicted .* emission_probability
        scale = sum(alpha)
        value += offset + log(scale)
        alpha = alpha ./ scale
    end
    return value
end


@model function hmm_example_model(N, K, y, initial)
    theta1 ~ Dirichlet(ones(K))
    theta2 ~ Dirichlet(ones(K))
    mu ~ ordered(product_distribution([
        truncated(Normal(3, 1); lower=0),
        truncated(Normal(10, 1); lower=0),
    ]))
    transition = [i == 1 ? theta1[j] : theta2[j] for i in 1:K, j in 1:K]
    emissions = NormalEmissions(mu, ones(K))
    y ~ HiddenMarkovModel(
        N, initial, transition, emissions, Val(:rows),
    )
end

function make_model(::Val{Symbol("hmm_example-hmm_example")}, data)
    y = reshape(Float64.(data["y"]), :, 1)
    return hmm_example_model(data["N"], data["K"], y, ones(data["K"]))
end


@model function hmm_drive_0(K, observations, alpha, initial)
    theta1 ~ Dirichlet(vec(alpha[1, :]))
    theta2 ~ Dirichlet(vec(alpha[2, :]))
    phi ~ ordered(product_distribution([
        truncated(Normal(0, 1); lower=0),
        truncated(Normal(3, 1); lower=0),
    ]))
    lambda ~ ordered(product_distribution([
        truncated(Normal(0, 1); lower=0),
        truncated(Normal(3, 1); lower=0),
    ]))
    transition = [i == 1 ? theta1[j] : theta2[j] for i in 1:K, j in 1:K]
    emissions = [
        product_distribution([
            Exponential(inv(phi[k])), Exponential(inv(lambda[k])),
        ])
        for k in 1:K
    ]
    observations ~ HiddenMarkovModel(
        size(observations, 1), initial, transition, emissions, Val(:rows),
    )
end

function make_model(::Val{Symbol("bball_drive_event_0-hmm_drive_0")}, data)
    observations = hcat(Float64.(data["u"]), Float64.(data["v"]))
    return hmm_drive_0(
        data["K"], observations, Float64.(data["alpha"]), ones(data["K"]),
    )
end


@model function hmm_drive_1(K, observations, alpha, tau, rho, initial)
    theta1 ~ Dirichlet(vec(alpha[1, :]))
    theta2 ~ Dirichlet(vec(alpha[2, :]))
    phi ~ ordered(product_distribution([Normal(0, 1), Normal(3, 1)]))
    lambda ~ ordered(product_distribution([Normal(0, 1), Normal(3, 1)]))
    transition = [i == 1 ? theta1[j] : theta2[j] for i in 1:K, j in 1:K]
    emissions = [
        product_distribution([
            Normal(phi[k], tau), Normal(lambda[k], rho),
        ])
        for k in 1:K
    ]
    observations ~ HiddenMarkovModel(
        size(observations, 1), initial, transition, emissions, Val(:rows),
    )
end

function make_model(::Val{Symbol("bball_drive_event_1-hmm_drive_1")}, data)
    observations = hcat(Float64.(data["u"]), Float64.(data["v"]))
    return hmm_drive_1(
        data["K"], observations, Float64.(data["alpha"]),
        Float64(data["tau"]), Float64(data["rho"]), ones(data["K"]),
    )
end


@model function lotka_volterra(N, t0, ts, y_init, fixed)
    data = fixed.value
    theta ~ product_distribution(Fill(FlatPos(0.0), 4))
    z_init ~ product_distribution(Fill(FlatPos(0.0), 2))
    sigma ~ product_distribution(Fill(LogNormal(-1, 1), 2))

    DynamicPPL.@addlogprob! logpdf(Normal(1, 0.5), theta[1])
    DynamicPPL.@addlogprob! logpdf(Normal(0.05, 0.05), theta[2])
    DynamicPPL.@addlogprob! logpdf(Normal(1, 0.5), theta[3])
    DynamicPPL.@addlogprob! logpdf(Normal(0.05, 0.05), theta[4])
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(LogNormal(log(10), 1), size(z_init))), z_init,
    )

    lotka_volterra_rhs! = function (derivative, populations, parameters, _)
        prey, predator = populations
        alpha, beta, gamma, delta = parameters
        derivative[1] = (alpha - beta * predator) * prey
        derivative[2] = (-gamma + delta * prey) * predator
        return nothing
    end
    z = _solve_ode(
        Tsit5(), lotka_volterra_rhs!, z_init, t0, ts, theta,
        1e-3, 1e-5, 500,
    )
    if minimum(z) <= 0
        DynamicPPL.@addlogprob! -Inf
        return
    end
    for species in 1:2
        y_init[species] ~ LogNormal(log(z_init[species]), sigma[species])
    end
    residuals = data.log_y_transposed .- log.(transpose(z))
    DynamicPPL.@addlogprob! loglikelihood(
        MvNormal(data.zero_mean, Diagonal(abs2.(sigma))), residuals,
    )
end

function make_model(::Val{Symbol("hudson_lynx_hare-lotka_volterra")}, data)
    fixed = _FixedData((
        log_y_transposed=permutedims(log.(Float64.(data["y"]))),
        zero_mean=zeros(2),
    ))
    return lotka_volterra(
        data["N"], _FixedData(0.0), _FixedData(Float64.(data["ts"])),
        Float64.(data["y_init"]), fixed,
    )
end


@model function one_comp_mm_elim_abs(t0, D, V, times, C_hat)
    k_a ~ truncated(Cauchy(0, 1); lower=0)
    K_m ~ truncated(Cauchy(0, 1); lower=0)
    V_m ~ truncated(Cauchy(0, 1); lower=0)
    sigma ~ truncated(Cauchy(0, 1); lower=0)
    parameters = [k_a, K_m, V_m, D, V]
    one_compartment_rhs! = function (derivative, concentration, parameters, time)
        absorption, michaelis, maximum_rate, dose_total, volume = parameters
        dose = time > 0 ?
            exp(-absorption * time) * dose_total * absorption / volume :
            zero(absorption)
        elimination = (maximum_rate / volume) * concentration[1] /
            (michaelis + concentration[1])
        derivative[1] = dose - elimination
        return nothing
    end
    concentration = _solve_ode(
        FBDF(), one_compartment_rhs!, [zero(k_a)], t0, times, parameters,
        1e-10, 1e-10, 100_000_000,
    )
    for n in eachindex(C_hat)
        if concentration[n, 1] <= 0
            DynamicPPL.@addlogprob! -Inf
            return
        end
        C_hat[n] ~ LogNormal(log(concentration[n, 1]), sigma)
    end
end

function make_model(::Val{Symbol("one_comp_mm_elim_abs-one_comp_mm_elim_abs")}, data)
    return one_comp_mm_elim_abs(
        _FixedData(Float64(data["t0"])), Float64(data["D"]), Float64(data["V"]),
        _FixedData(Float64.(data["times"])), Float64.(data["C_hat"]),
    )
end


@model function accel_gp(
    Y, Xgp_1, slambda_sq_1, slambda_dimension_1,
    Xgp_sigma_1, slambda_sigma_sq_1, slambda_sigma_dimension_1,
    NBgp_1, NBgp_sigma_1, prior_only,
)
    Intercept ~ LocationScale(-13, 36, TDist(3))
    sdgp_1 ~ truncated(LocationScale(0, 36, TDist(3)); lower=0)
    lscale_1 ~ InverseGamma(1.124909, 0.0177)
    zgp_1 ~ product_distribution(Fill(Normal(), NBgp_1))
    Intercept_sigma ~ LocationScale(0, 10, TDist(3))
    sdgp_sigma_1 ~ truncated(LocationScale(0, 36, TDist(3)); lower=0)
    lscale_sigma_1 ~ InverseGamma(1.124909, 0.0177)
    zgp_sigma_1 ~ product_distribution(Fill(Normal(), NBgp_sigma_1))

    gp_constant = sdgp_1 *
        (sqrt(2pi) * lscale_1)^(slambda_dimension_1 / 2)
    gp_scale = gp_constant .* exp.(-0.25 * lscale_1^2 .* slambda_sq_1)
    mu = Intercept .+ Xgp_1 * (gp_scale .* zgp_1)
    sigma_constant = sdgp_sigma_1 *
        (sqrt(2pi) * lscale_sigma_1)^(slambda_sigma_dimension_1 / 2)
    sigma_scale = sigma_constant .* exp.(
        -0.25 * lscale_sigma_1^2 .* slambda_sigma_sq_1,
    )
    sigma = exp.(Intercept_sigma .+ Xgp_sigma_1 * (sigma_scale .* zgp_sigma_1))
    if prior_only == 0
        Y ~ MvNormal(mu, Diagonal(abs2.(sigma)))
    end
end

function make_model(::Val{Symbol("mcycle_gp-accel_gp")}, data)
    slambda_1 = Float64.(data["slambda_1"])
    slambda_sigma_1 = Float64.(data["slambda_sigma_1"])
    return accel_gp(
        Float64.(data["Y"]), Float64.(data["Xgp_1"]),
        vec(sum(abs2, slambda_1; dims=2)), size(slambda_1, 2),
        Float64.(data["Xgp_sigma_1"]),
        vec(sum(abs2, slambda_sigma_1; dims=2)), size(slambda_sigma_1, 2),
        data["NBgp_1"], data["NBgp_sigma_1"], data["prior_only"],
    )
end













































# PosteriorDB Stan model: 2pl_latent_reg_irt
@model function pdb_2pl_latent_reg_irt(I, J, N, ii, jj, y, W_adj)
    K = size(W_adj, 2)
    alpha ~ product_distribution(Fill(FlatPos(0), I))
    beta_free ~ product_distribution(Fill(Flat(), I - 1))
    theta ~ product_distribution(Fill(Flat(), J))
    lambda_adj ~ product_distribution(Fill(Flat(), K))
    beta = vcat(beta_free, -(sum(beta_free)))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(LogNormal(1, 1), size(alpha))), alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 3), size(beta))), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(LocationScale(0, 1, TDist(3)), size(lambda_adj))), lambda_adj)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(), size(theta))),
        theta .- W_adj * lambda_adj,
    )
    logits = alpha[ii] .* theta[jj] .- beta[ii]
    DynamicPPL.@addlogprob! -sum(log1pexp.(y .* logits))
end

function make_model(::Val{Symbol("fims_Aus_Jpn_irt-2pl_latent_reg_irt")}, data)
    W = Float64.(data["W"])
    adjustment = zeros(2, size(W, 2))
    adjustment[2, 1] = 1
    for k in 2:size(W, 2)
        # Preserve the published Stan program's effective two-s.d. branch.
        adjustment[1, k] = mean(view(W, :, k))
        adjustment[2, k] = 2std(view(W, :, k))
    end
    W_adj = (W .- adjustment[1:1, :]) ./ adjustment[2:2, :]
    return pdb_2pl_latent_reg_irt(
        data["I"], data["J"], data["N"], data["ii"], data["jj"],
        1 .- 2 .* Float64.(data["y"]), W_adj,
    )
end



# PosteriorDB Stan model: GLMM1_model
@model function pdb_GLMM1_model(nsite, nobs, obs, obsyear, obssite, misyear, missite)
    alpha ~ product_distribution(Fill(Flat(), nsite))
    mu_alpha ~ Flat()
    sd_alpha ~ Uniform(0, 5)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sd_alpha), size(alpha))), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), alpha[obssite])), obs,
    )
end

function make_model(::Val{Symbol("GLMM_data-GLMM1_model")}, data)
    return pdb_GLMM1_model(data["nsite"], data["nobs"], data["obs"], data["obsyear"], data["obssite"], data["misyear"], data["missite"])
end



# PosteriorDB Stan model: GLMM_Poisson_model
@model function pdb_GLMM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 20)
    beta3 ~ Uniform(-10, 10)
    eps ~ product_distribution(Fill(Flat(), n))
    sigma ~ Uniform(0, 5)
    log_lambda = Base.materialize(Base.broadcasted(+, alpha, Base.broadcasted(*, beta1, year), Base.broadcasted(*, beta2, year_squared), Base.broadcasted(*, beta3, year_cubed), eps))
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), log_lambda)), C,
    )
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma), size(eps))), eps)
end

function make_model(::Val{Symbol("GLMM_Poisson_data-GLMM_Poisson_model")}, data)
    year = data["year"]
    return pdb_GLMM_Poisson_model(
        data["n"], data["C"], year, year .^ 2, year .^ 3,
    )
end



# PosteriorDB Stan model: GLM_Binomial_model
@model function pdb_GLM_Binomial_model(nyears, C, N, year, year_squared)
    alpha ~ Flat()
    beta1 ~ Flat()
    beta2 ~ Flat()
    logit_p = Base.materialize(Base.broadcasted(+, alpha, Base.broadcasted(*, beta1, year), Base.broadcasted(*, beta2, year_squared)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 100), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 100), beta1)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 100), beta2)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(BinomialLogit, N, logit_p)), C,
    )
end

function make_model(::Val{Symbol("GLM_Binomial_data-GLM_Binomial_model")}, data)
    year = data["year"]
    return pdb_GLM_Binomial_model(
        data["nyears"], data["C"], data["N"], year, year .^ 2,
    )
end



# PosteriorDB Stan model: GLM_Poisson_model
@model function pdb_GLM_Poisson_model(n, C, year, year_squared, year_cubed)
    alpha ~ Uniform(-20, 20)
    beta1 ~ Uniform(-10, 10)
    beta2 ~ Uniform(-10, 10)
    beta3 ~ Uniform(-10, 10)
    log_lambda = Base.materialize(Base.broadcasted(+, alpha, Base.broadcasted(*, beta1, year), Base.broadcasted(*, beta2, year_squared), Base.broadcasted(*, beta3, year_cubed)))
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(x -> Poisson(exp(x)), log_lambda)), C,
    )
end

function make_model(::Val{Symbol("GLM_Poisson_Data-GLM_Poisson_model")}, data)
    year = data["year"]
    return pdb_GLM_Poisson_model(
        data["n"], data["C"], year, year .^ 2, year .^ 3,
    )
end



# PosteriorDB Stan model: M0_model
@model function pdb_M0_model(M, T, y, s)
    @assert size(y) == (M, T)
    @assert size(s) == (M,)
    omega ~ Uniform(0, 1)
    p ~ Uniform(0, 1)
    for i = 1:M
        if s[i] > 0
            DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) + logpdf(Binomial(T, p), s[i])
        else
            DynamicPPL.@addlogprob! logaddexp(
                logpdf(Bernoulli(omega), 1) + logpdf(Binomial(T, p), 0),
                logpdf(Bernoulli(omega), 0),
            )
        end
    end
end

function make_model(::Val{Symbol("M0_data-M0_model")}, data)
    y = data["y"]
    return pdb_M0_model(data["M"], data["T"], y, vec(sum(y; dims=2)))
end



# PosteriorDB Stan model: Mb_model
@model function pdb_Mb_model(
    T, captured, unobserved, p_success, p_failure, c_success, c_failure,
)
    omega ~ Uniform(0.0, 1.0)
    p ~ Uniform(0.0, 1.0)
    c ~ Uniform(0.0, 1.0)
    DynamicPPL.@addlogprob! captured * log(omega) +
        p_success * log(p) + p_failure * log1p(-p) +
        c_success * log(c) + c_failure * log1p(-c) +
        unobserved * logaddexp(log(omega) + T * log1p(-p), log1p(-omega))
end

function make_model(::Val{Symbol("Mb_data-Mb_model")}, data)
    y = Int.(data["y"])
    captured = 0
    p_success = p_failure = c_success = c_failure = 0
    for i in axes(y, 1)
        observed = sum(view(y, i, :)) > 0
        captured += observed
        observed || continue
        p_success += y[i, 1]
        p_failure += 1 - y[i, 1]
        for t in 2:size(y, 2)
            if y[i, t - 1] == 0
                p_success += y[i, t]
                p_failure += 1 - y[i, t]
            else
                c_success += y[i, t]
                c_failure += 1 - y[i, t]
            end
        end
    end
    return pdb_Mb_model(
        data["T"], captured, data["M"] - captured,
        p_success, p_failure, c_success, c_failure,
    )
end



# PosteriorDB Stan model: Mh_model
@model function pdb_Mh_model(M, T, y)
    @assert size(y) == (M,)
    omega ~ Uniform(0, 1)
    mean_p ~ Uniform(0, 1)
    sigma ~ Uniform(0, 5)
    eps_raw ~ product_distribution(Fill(Flat(), M))
    eps = Base.materialize(Base.broadcasted(+, Base.broadcasted(logit, mean_p), Base.broadcasted(*, sigma, eps_raw)))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(eps_raw))), eps_raw)
    for i = 1:M
        if y[i] > 0
            DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) + logpdf(BinomialLogit(T, eps[i]), y[i])
        else
            DynamicPPL.@addlogprob! logaddexp(
                logpdf(Bernoulli(omega), 1) +
                    logpdf(BinomialLogit(T, eps[i]), 0),
                logpdf(Bernoulli(omega), 0),
            )
        end
    end
end

function make_model(::Val{Symbol("Mh_data-Mh_model")}, data)
    return pdb_Mh_model(data["M"], data["T"], data["y"])
end



# PosteriorDB Stan model: Mt_model
@model function pdb_Mt_model(M, T, y, s)
    @assert size(y) == (M, T)
    @assert size(s) == (M,)
    omega ~ Uniform(0, 1)
    p ~ product_distribution(Fill(Uniform(0, 1), T))
            for i = 1:M
                if s[i] > 0
                    DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) + logpdf(product_distribution(map(Bernoulli, p)), y[i, :])
                else
                    DynamicPPL.@addlogprob! logaddexp(
                        logpdf(Bernoulli(omega), 1) +
                            logpdf(product_distribution(map(Bernoulli, p)), y[i, :]),
                        logpdf(Bernoulli(omega), 0),
                    )
                end
            end
end

function make_model(::Val{Symbol("Mt_data-Mt_model")}, data)
    y = data["y"]
    return pdb_Mt_model(data["M"], data["T"], y, vec(sum(y; dims=2)))
end



# PosteriorDB Stan model: Mtbh_model
@model function pdb_Mtbh_model(M, T, y, s)
    @assert size(y) == (M, T)
    @assert size(s) == (M,)
    omega ~ Uniform(0, 1)
    mean_p ~ product_distribution(Fill(Uniform(0, 1), T))
    gamma ~ Flat()
    sigma ~ Uniform(0, 3)
    eps_raw ~ product_distribution(Fill(Flat(), M))
            eps = Base.materialize(Base.broadcasted(*, sigma, eps_raw))
            alpha = Base.materialize(Base.broadcasted(logit, mean_p))
            logit_p = hcat(Base.materialize(Base.broadcasted(+, alpha[1], eps)), Base.materialize(Base.broadcasted(+, (alpha[2:end])', eps, Base.broadcasted(*, gamma, y[:, 1:end - 1]))))
            DynamicPPL.@addlogprob! logpdf(Normal(0, 10), gamma)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(eps_raw))), eps_raw)
            for i = 1:M
                if s[i] > 0
                    DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) + logpdf(
                        product_distribution(map(BernoulliLogit, logit_p[i, :])), y[i, :],
                    )
                else
                    DynamicPPL.@addlogprob! logaddexp(
                        logpdf(Bernoulli(omega), 1) + logpdf(
                            product_distribution(map(BernoulliLogit, logit_p[i, :])), y[i, :],
                        ),
                        logpdf(Bernoulli(omega), 0),
                    )
                end
            end
end

function make_model(::Val{Symbol("Mtbh_data-Mtbh_model")}, data)
    y = data["y"]
    return pdb_Mtbh_model(data["M"], data["T"], y, vec(sum(y; dims=2)))
end



# PosteriorDB Stan model: Mth_model
@model function pdb_Mth_model(M, T, y, s)
    @assert size(y) == (M, T)
    @assert size(s) == (M,)
    omega ~ Uniform(0.0, 1.0)
    mean_p ~ product_distribution(Fill(Uniform(0.0, 1.0), T))
    sigma ~ Uniform(0.0, 5.0)
    eps_raw ~ product_distribution(Fill(Flat(), M))
            logit_p = logit.(mean_p)' .+ sigma .* eps_raw
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, 1.0), size(eps_raw))), eps_raw)
            for i = 1:M
                if s[i] > 0
                    DynamicPPL.@addlogprob! logpdf(Bernoulli(omega), 1) + logpdf(
                        product_distribution(map(BernoulliLogit, logit_p[i, :])), y[i, :],
                    )
                else
                    DynamicPPL.@addlogprob! logaddexp(
                        logpdf(Bernoulli(omega), 1) + logpdf(
                            product_distribution(map(BernoulliLogit, logit_p[i, :])), y[i, :],
                        ),
                        logpdf(Bernoulli(omega), 0),
                    )
                end
            end
end

function make_model(::Val{Symbol("Mth_data-Mth_model")}, data)
    y = data["y"]
    return pdb_Mth_model(data["M"], data["T"], y, vec(sum(y; dims=2)))
end



# PosteriorDB Stan model: Rate_1_model
@model function pdb_Rate_1_model(n, k)
    theta ~ Beta(1, 1)
    k ~ Binomial(n, theta)
end

function make_model(::Val{Symbol("Rate_1_data-Rate_1_model")}, data)
    return pdb_Rate_1_model(data["n"], data["k"])
end



# PosteriorDB Stan model: Rate_2_model
@model function pdb_Rate_2_model(n1, n2, k1, k2)
    theta1 ~ Beta(1, 1)
    theta2 ~ Beta(1, 1)
    delta = theta1 - theta2
    k1 ~ Binomial(n1, theta1)
    k2 ~ Binomial(n2, theta2)
end

function make_model(::Val{Symbol("Rate_2_data-Rate_2_model")}, data)
    return pdb_Rate_2_model(data["n1"], data["n2"], data["k1"], data["k2"])
end



# PosteriorDB Stan model: Rate_3_model
@model function pdb_Rate_3_model(n1, n2, k1, k2)
    theta ~ Beta(1, 1)
    k1 ~ Binomial(n1, theta)
    k2 ~ Binomial(n2, theta)
end

function make_model(::Val{Symbol("Rate_3_data-Rate_3_model")}, data)
    return pdb_Rate_3_model(data["n1"], data["n2"], data["k1"], data["k2"])
end



# PosteriorDB Stan model: Rate_4_model
@model function pdb_Rate_4_model(n, k)
    theta ~ Beta(1, 1)
    thetaprior ~ Beta(1, 1)
    k ~ Binomial(n, theta)
end

function make_model(::Val{Symbol("Rate_4_data-Rate_4_model")}, data)
    return pdb_Rate_4_model(data["n"], data["k"])
end



# PosteriorDB Stan model: Rate_5_model
@model function pdb_Rate_5_model(n1, n2, k1, k2)
    theta ~ Beta(1, 1)
    k1 ~ Binomial(n1, theta)
    k2 ~ Binomial(n2, theta)
end

function make_model(::Val{Symbol("Rate_5_data-Rate_5_model")}, data)
    return pdb_Rate_5_model(data["n1"], data["n2"], data["k1"], data["k2"])
end



# PosteriorDB Stan model: Survey_model
@model function pdb_Survey_model(successes, failures, log_coefficients)
    theta ~ Uniform(0, 1)
    terms = log_coefficients .+ successes * log(theta) .+
        failures .* log1p(-theta)
    DynamicPPL.@addlogprob! logsumexp(terms)
end

function make_model(::Val{Symbol("Survey_data-Survey_model")}, data)
    nmax = data["nmax"]
    m = data["m"]
    k = data["k"]
    successes = sum(k)
    ns = collect(maximum(k):nmax)
    log_coefficients = [
        -log(nmax) + sum(ki -> logpdf(Binomial(n, 0.5), ki), k) -
        m * n * log(0.5)
        for n in ns
    ]
    failures = m .* ns .- successes
    return pdb_Survey_model(successes, failures, log_coefficients)
end



# PosteriorDB Stan model: accel_splines
@model function pdb_accel_splines(N, Y, Ks, Xs, knots_1, Zs_1_1, Ks_sigma, Xs_sigma, knots_sigma_1, Zs_sigma_1_1, prior_only)
    Y = Y.value
    Xs = Xs.value
    Zs_1_1 = Zs_1_1.value
    Xs_sigma = Xs_sigma.value
    Zs_sigma_1_1 = Zs_sigma_1_1.value
    Intercept ~ Flat()
    bs ~ product_distribution(Fill(Flat(), Ks))
    zs_1_1 ~ product_distribution(Fill(Flat(), knots_1))
    sds_1_1 ~ FlatPos(0)
    Intercept_sigma ~ Flat()
    bs_sigma ~ product_distribution(Fill(Flat(), Ks_sigma))
    zs_sigma_1_1 ~ product_distribution(Fill(Flat(), knots_sigma_1))
    sds_sigma_1_1 ~ FlatPos(0)
    s_1_1 = Base.materialize(Base.broadcasted(*, sds_1_1, zs_1_1))
    s_sigma_1_1 = Base.materialize(Base.broadcasted(*, sds_sigma_1_1, zs_sigma_1_1))
    mu = (Intercept .+ Xs * bs) + Zs_1_1 * s_1_1
    log_sigma = (Intercept_sigma .+ Xs_sigma * bs_sigma) +
        Zs_sigma_1_1 * s_sigma_1_1
    DynamicPPL.@addlogprob! logpdf(LocationScale(-13, 36, TDist(3)), Intercept)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(zs_1_1))), zs_1_1)
    DynamicPPL.@addlogprob! logpdf(LocationScale(0, 36, TDist(3)), sds_1_1)
    DynamicPPL.@addlogprob! logpdf(LocationScale(0, 10, TDist(3)), Intercept_sigma)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(zs_sigma_1_1))), zs_sigma_1_1)
    DynamicPPL.@addlogprob! logpdf(LocationScale(0, 36, TDist(3)), sds_sigma_1_1)
    if !(prior_only == 1)
        sigma = exp.(log_sigma)
        DynamicPPL.@addlogprob! logpdf(MvNormal(mu, Diagonal(abs2.(sigma))), Y)
    end
end

function make_model(::Val{Symbol("mcycle_splines-accel_splines")}, data)
    return pdb_accel_splines(
        data["N"], _FixedData(Float64.(data["Y"])), data["Ks"],
        _FixedData(Float64.(data["Xs"])), data["knots_1"],
        _FixedData(Float64.(data["Zs_1_1"])), data["Ks_sigma"],
        _FixedData(Float64.(data["Xs_sigma"])), data["knots_sigma_1"],
        _FixedData(Float64.(data["Zs_sigma_1_1"])), data["prior_only"],
    )
end



# PosteriorDB Stan model: bones_model
@model function pdb_bones_model(nChild, nInd, gamma, delta, ncat, grade)
    theta ~ product_distribution(Fill(Flat(), nChild))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, 36.0), size(theta))), theta)
    p = zeros(typeof(theta[1]), nChild, nInd, 5)
    Q = zeros(typeof(theta[1]), nChild, nInd, 4)
    for i = 1:nChild
        for j = 1:nInd
            for k = 1:ncat[j] - 1
                Q[i, j, k] = logistic(delta[j] * (theta[i] - gamma[j, k]))
            end
            p[i, j, 1] = 1 - Q[i, j, 1]
            for k = 2:ncat[j] - 1
                p[i, j, k] = Q[i, j, k - 1] - Q[i, j, k]
            end
            p[i, j, ncat[j]] = Q[i, j, ncat[j] - 1]
            if grade[i, j] != -1
                DynamicPPL.@addlogprob! log(p[i, j, grade[i, j]])
            end
        end
    end
end

function make_model(::Val{Symbol("bones_data-bones_model")}, data)
    return pdb_bones_model(data["nChild"], data["nInd"], data["gamma"], data["delta"], data["ncat"], data["grade"])
end



# PosteriorDB Stan model: bym2_offset_only
@model function pdb_bym2_offset_only(
    N, N_edges, node1, node2, y, log_E, scaling_factor,
)
    beta0 ~ Flat()
    sigma ~ FlatPos(0)
    rho ~ Uniform(0, 1)
    theta ~ product_distribution(Fill(Flat(), N))
    phi ~ product_distribution(Fill(Flat(), N))
    convolved_re = Base.materialize(Base.broadcasted(+, Base.broadcasted(*, Base.broadcasted(sqrt, Base.broadcasted(-, 1, rho)), theta), Base.broadcasted(*, Base.broadcasted(sqrt, Base.broadcasted(/, rho, scaling_factor)), phi)))
            log_rate = log_E .+ beta0 .+ convolved_re .* sigma
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(map(x -> Poisson(exp(x)), log_rate)), y,
            )
            DynamicPPL.@addlogprob! -0.5 * sum(abs2, phi[node1] - phi[node2])
            DynamicPPL.@addlogprob! logpdf(Normal(0, 1), beta0)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(theta))), theta)
            DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma)
            DynamicPPL.@addlogprob! logpdf(Beta(0.5, 0.5), rho)
            DynamicPPL.@addlogprob! logpdf(Normal(0, 0.001N), sum(phi))
end

function make_model(::Val{Symbol("traffic_accident_nyc-bym2_offset_only")}, data)
    return pdb_bym2_offset_only(
        data["N"], data["N_edges"], data["node1"], data["node2"], data["y"],
        log.(data["E"]), data["scaling_factor"],
    )
end



# PosteriorDB Stan model: covid19imperial_v2
@model function pdb_covid19imperial_v2(M, P, fixed, observed_deaths)
    data = fixed.value
    mu ~ product_distribution(Fill(FlatPos(0), M))
    alpha_hier ~ product_distribution(Fill(Gamma(0.1667, 1), P))
    kappa ~ truncated(Normal(0, 0.5); lower=0)
    tau ~ Exponential(inv(0.03))
    y ~ product_distribution(Fill(Exponential(tau), M))
    phi ~ truncated(Normal(0, 5); lower=0)
    ifr_noise ~ product_distribution(Fill(truncated(Normal(1, 0.1); lower=0), M))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(3.28, kappa), size(mu))), mu)

    death_mean_parts = Vector{Vector{typeof(mu[1])}}(undef, M)
    adjusted_alpha = alpha_hier .- log(1.05) / 6
    for m in 1:M
        prediction = zeros(typeof(mu[1]), data.N2)
        prediction[1:data.N0] .= y[m]
        cumulative = zero(mu[1])
        for i in 2:data.N0
            cumulative += prediction[i]
        end
        for i in (data.N0 + 1):data.N2
            infection = dot(data.SI_weights[i], prediction)
            cumulative += prediction[i - 1]
            reproduction = mu[m] * exp(
                -dot(view(data.X, m, i, :), adjusted_alpha),
            )
            susceptible = (data.pop[m] - cumulative) / data.pop[m]
            prediction[i] = reproduction * susceptible * infection
        end
        expected_deaths = ifr_noise[m] .* (data.death_weights[m] * prediction)
        expected_deaths[1] = 1.0e-15 * prediction[1]
        death_mean_parts[m] = expected_deaths[data.EpidemicStart[m]:data.N[m]]
    end
    death_means = reduce(vcat, death_mean_parts)
    log_denominator = log.(phi .+ death_means)
    DynamicPPL.@addlogprob! sum(loggamma.(observed_deaths .+ phi)) -
        length(observed_deaths) * loggamma(phi) - data.count_logfactorial +
        length(observed_deaths) * phi * log(phi) -
        phi * sum(log_denominator) +
        sum(xlogy.(observed_deaths, death_means)) -
        dot(observed_deaths, log_denominator)
end

function _make_covid(data)
    f = Float64.(data["f"])
    N2 = Int(data["N2"])
    SI_rev = reverse(Float64.(data["SI"]))
    f_rev = permutedims(mapreduce(reverse, hcat, eachcol(f)))
    SI_weights = [zeros(N2) for _ in 1:N2]
    for i in 2:N2, j in 1:(i - 1)
        SI_weights[i][j] = SI_rev[length(SI_rev) - i + j + 1]
    end
    death_weights = [zeros(N2, N2) for _ in 1:data["M"]]
    for m in 1:data["M"], i in 2:N2, j in 1:(i - 1)
        death_weights[m][i, j] = f_rev[m, size(f_rev, 2) - i + j + 1]
    end
    deaths = Int.(data["deaths"])
    epidemic_start = Int.(data["EpidemicStart"])
    observation_end = Int.(data["N"])
    observed_deaths = [
        deaths[i, m]
        for m in 1:data["M"]
        for i in epidemic_start[m]:observation_end[m]
    ]
    fixed = _FixedData((
        N0=Int(data["N0"]),
        N2=N2,
        N=observation_end,
        EpidemicStart=epidemic_start,
        X=Float64.(data["X"]),
        pop=Float64.(data["pop"]),
        SI_weights=SI_weights,
        death_weights=death_weights,
        count_logfactorial=sum(loggamma.(observed_deaths .+ 1)),
    ))
    return pdb_covid19imperial_v2(
        data["M"], data["P"], fixed, observed_deaths,
    )
end

make_model(::Val{Symbol("ecdc0401-covid19imperial_v2")}, data) = _make_covid(data)

make_model(::Val{Symbol("ecdc0401-covid19imperial_v3")}, data) = _make_covid(data)

make_model(::Val{Symbol("ecdc0501-covid19imperial_v2")}, data) = _make_covid(data)

make_model(::Val{Symbol("ecdc0501-covid19imperial_v3")}, data) = _make_covid(data)



# PosteriorDB Stan model: dogs
@model function pdb_dogs(n_dogs, n_trials, y, prev_shock, prev_avoid)
    beta ~ product_distribution(Fill(Flat(), 3))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 100), size(beta))), beta)
    for i = 1:n_dogs
        for j = 1:n_trials
            p = beta[1] + beta[2] * prev_avoid[i, j] + beta[3] * prev_shock[i, j]
            DynamicPPL.@addlogprob! logpdf(BernoulliLogit(p), y[i, j])
        end
    end
end

function make_model(::Val{Symbol("dogs-dogs")}, data)
    n_dogs, n_trials, y = data["n_dogs"], data["n_trials"], data["y"]
    past = y[:, 1:(n_trials - 1)]
    prev_shock = hcat(zeros(n_dogs), cumsum(past; dims=2))
    prev_avoid = hcat(zeros(n_dogs), cumsum(1 .- past; dims=2))
    return pdb_dogs(n_dogs, n_trials, y, prev_shock, prev_avoid)
end



# PosteriorDB Stan model: dogs_hierarchical
@model function pdb_dogs_hierarchical(
    n_dogs, n_trials, y, prev_shock, prev_avoid,
)
    J = n_dogs
    T = n_trials
    a ~ Uniform(0, 1)
    b ~ Uniform(0, 1)
    DynamicPPL.@addlogprob! logpdf(product_distribution(map(Bernoulli, Base.materialize(Base.broadcasted(*, Base.broadcasted(^, a, prev_shock), Base.broadcasted(^, b, prev_avoid))))), y)
end

function make_model(::Val{Symbol("dogs-dogs_hierarchical")}, data)
    n_dogs, n_trials, y = data["n_dogs"], data["n_trials"], data["y"]
    past = y[:, 1:(n_trials - 1)]
    prev_shock = hcat(zeros(n_dogs), cumsum(past; dims=2))
    prev_avoid = hcat(zeros(n_dogs), cumsum(1 .- past; dims=2))
    return pdb_dogs_hierarchical(
        n_dogs, n_trials, y, prev_shock, prev_avoid,
    )
end



# PosteriorDB Stan model: dogs_log
@model function pdb_dogs_log(n_dogs, n_trials, y, prev_shock, prev_avoid)
    @assert size(y) == (n_dogs, n_trials)
    # Stan declares beta unconstrained and applies bounded priors in the model.
    # Keep the same identity parameterisation rather than introducing DynamicPPL's
    # bounded transform for a product of Uniform distributions.
    beta ~ product_distribution(Fill(Flat(), 2))
    DynamicPPL.@addlogprob! logpdf(Uniform(-100, 0), beta[1])
    DynamicPPL.@addlogprob! logpdf(Uniform(0, 100), beta[2])
    for i = 1:n_dogs
        for j = 1:n_trials
            p = logistic(
                beta[1] * prev_avoid[i, j] + beta[2] * prev_shock[i, j],
            )
            DynamicPPL.@addlogprob! logpdf(Bernoulli(p), y[i, j])
        end
    end
end

function make_model(::Val{Symbol("dogs-dogs_log")}, data)
    n_dogs, n_trials, y = data["n_dogs"], data["n_trials"], data["y"]
    past = y[:, 1:(n_trials - 1)]
    prev_shock = hcat(zeros(n_dogs), cumsum(past; dims=2))
    prev_avoid = hcat(zeros(n_dogs), cumsum(1 .- past; dims=2))
    return pdb_dogs_log(n_dogs, n_trials, y, prev_shock, prev_avoid)
end



# PosteriorDB Stan model: dogs_nonhierarchical
@model function pdb_dogs_nonhierarchical(
    n_dogs, n_trials, y, prev_shock, prev_avoid,
)
    J = n_dogs
    T = n_trials
    mu_logit_ab ~ product_distribution(Fill(Flat(), 2))
    sigma_logit_ab ~ product_distribution(Fill(FlatPos(0), 2))
    L_logit_ab ~ LKJCholesky(2, 2.0)
    z ~ product_distribution(Fill(Flat(), J, 2))
            logit_ab = ones(J) * mu_logit_ab' + z * (Diagonal(sigma_logit_ab) * L_logit_ab.L)
            a = logistic.(logit_ab[:, 1])
            b = logistic.(logit_ab[:, 2])
            DynamicPPL.@addlogprob! logpdf(product_distribution(map(Bernoulli, Base.materialize(Base.broadcasted(*, Base.broadcasted(^, a, prev_shock), Base.broadcasted(^, b, prev_avoid))))), y)
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Logistic(0, 1), size(mu_logit_ab))),
                mu_logit_ab,
            )
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(sigma_logit_ab))), sigma_logit_ab)
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Normal(0, 1), length(z))), vec(z),
            )
end

function make_model(::Val{Symbol("dogs-dogs_nonhierarchical")}, data)
    n_dogs, n_trials, y = data["n_dogs"], data["n_trials"], data["y"]
    past = y[:, 1:(n_trials - 1)]
    prev_shock = hcat(zeros(n_dogs), cumsum(past; dims=2))
    prev_avoid = hcat(zeros(n_dogs), cumsum(1 .- past; dims=2))
    return pdb_dogs_nonhierarchical(
        n_dogs, n_trials, y, prev_shock, prev_avoid,
    )
end



# PosteriorDB Stan model: dugongs_model
@model function pdb_dugongs_model(N, x, Y)
    x_ = x
    alpha ~ Flat()
    beta ~ Flat()
    lambda ~ Uniform(0.5, 1.0)
    tau ~ FlatPos(0.0)
    sigma = 1.0 / sqrt(tau)
    U3 = logit(lambda)
    for i = 1:N
        m = alpha - beta * lambda^x_[i]
        DynamicPPL.@addlogprob! logpdf(Normal(m, sigma), Y[i])
    end
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), beta)
    DynamicPPL.@addlogprob! logpdf(Gamma(0.0001, inv(0.0001)), tau)
end

function make_model(::Val{Symbol("dugongs_data-dugongs_model")}, data)
    return pdb_dugongs_model(data["N"], data["x"], data["Y"])
end



# PosteriorDB Stan model: election88_full
@model function pdb_election88_full(N, n_age, n_age_edu, n_edu, n_region_full, n_state, age, age_edu, black, edu, female, region_full, state, v_prev_full, y)
    a ~ product_distribution(Fill(Flat(), n_age))
    b ~ product_distribution(Fill(Flat(), n_edu))
    c ~ product_distribution(Fill(Flat(), n_age_edu))
    d ~ product_distribution(Fill(Flat(), n_state))
    e ~ product_distribution(Fill(Flat(), n_region_full))
    beta ~ product_distribution(Fill(Flat(), 5))
    sigma_a ~ Uniform(0, 100)
    sigma_b ~ Uniform(0, 100)
    sigma_c ~ Uniform(0, 100)
    sigma_d ~ Uniform(0, 100)
    sigma_e ~ Uniform(0, 100)
            y_hat = Base.materialize(Base.broadcasted(+, beta[1], Base.broadcasted(*, beta[2], black), Base.broadcasted(*, beta[3], female), Base.broadcasted(*, beta[5], female, black), Base.broadcasted(*, beta[4], v_prev_full), a[age], b[edu], c[age_edu], d[state], e[region_full]))
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_a), size(a))), a)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_b), size(b))), b)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_c), size(c))), c)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_d), size(d))), d)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_e), size(e))), e)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 100), size(beta))), beta)
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(map(BernoulliLogit, y_hat)), y,
            )
end

function make_model(::Val{Symbol("election88-election88_full")}, data)
    return pdb_election88_full(data["N"], data["n_age"], data["n_age_edu"], data["n_edu"], data["n_region_full"], data["n_state"], data["age"], data["age_edu"], data["black"], data["edu"], data["female"], data["region_full"], data["state"], data["v_prev_full"], data["y"])
end



# PosteriorDB Stan model: gpcm_latent_reg_irt
@model function pdb_gpcm_latent_reg_irt(
    I, J, N, ii, jj, response, m, W_adj, levels, valid,
    response_indices, step_design,
)
    K = size(W_adj, 2)
    alpha ~ product_distribution(Fill(LogNormal(1, 1), I))
    beta_free ~ product_distribution(Fill(Flat(), sum(m) - 1))
    lambda_adj ~ product_distribution(Fill(LocationScale(0, 1, TDist(3)), K))
    theta ~ MvNormal(W_adj * lambda_adj, 1.0)
    beta = vcat(beta_free, -(sum(beta_free)))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 3), size(beta))), beta)
    ability = alpha[ii] .* theta[jj]
    cumulative_difficulty = reshape(
        step_design * beta, I, size(levels, 2),
    )
    logits = reshape(ability, N, 1) .* levels .-
        cumulative_difficulty[ii, :]
    logits = ifelse.(valid, logits, -Inf)
    DynamicPPL.@addlogprob! sum(logits[response_indices]) -
        sum(logsumexp(logits; dims=2))
end

function make_model(::Val{Symbol("timssAusTwn_irt-gpcm_latent_reg_irt")}, data)
    I = data["I"]
    ii = Int.(data["ii"])
    y = Int.(data["y"])
    m = zeros(Int, I)
    for n in eachindex(y)
        m[ii[n]] = max(m[ii[n]], y[n])
    end
    pos = ones(Int, I)
    for i in 2:I
        pos[i] = m[i - 1] + pos[i - 1]
    end
    W = Float64.(data["W"])
    adjustment = zeros(2, size(W, 2))
    adjustment[2, 1] = 1
    for k in 2:size(W, 2)
        adjustment[1, k] = mean(view(W, :, k))
        adjustment[2, k] = 2std(view(W, :, k))
    end
    W_adj = (W .- adjustment[1:1, :]) ./ adjustment[2:2, :]
    max_level = maximum(m)
    levels = reshape(Float64.(0:max_level), 1, :)
    valid = [level <= m[ii[n]] for n in 1:data["N"], level in 0:max_level]
    response = y .+ 1
    response_indices = [
        n + (response[n] - 1) * data["N"] for n in 1:data["N"]
    ]
    step_design = zeros(I * (max_level + 1), sum(m))
    for item in 1:I, level in 1:m[item], step in 1:level
        row = item + level * I
        step_design[row, pos[item] + step - 1] = 1.0
    end
    return pdb_gpcm_latent_reg_irt(
        I, data["J"], data["N"], ii, data["jj"], response, m, W_adj,
        levels, valid, response_indices, step_design,
    )
end



# PosteriorDB Stan model: grsm_latent_reg_irt
@model function pdb_grsm_latent_reg_irt(
    I, J, N, ii, jj, response, max_category, W_adj, levels,
    response_indices,
)
    m = first(max_category)
    K = size(W_adj, 2)
    alpha ~ product_distribution(Fill(LogNormal(1, 1), I))
    beta_free ~ product_distribution(Fill(Flat(), I - 1))
    kappa_free ~ product_distribution(Fill(Flat(), m - 1))
    lambda_adj ~ product_distribution(Fill(LocationScale(0, 1, TDist(3)), K))
    theta ~ MvNormal(W_adj * lambda_adj, 1.0)
    beta = vcat(beta_free, -(sum(beta_free)))
    kappa = vcat(kappa_free, -(sum(kappa_free)))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 3), size(beta))), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 3), size(kappa))), kappa)
    ability = alpha[ii] .* theta[jj] .- beta[ii]
    cumulative_kappa = vcat(zero(kappa[1]), cumsum(kappa))
    logits = reshape(ability, N, 1) .* levels .-
        reshape(cumulative_kappa, 1, :)
    DynamicPPL.@addlogprob! sum(logits[response_indices]) -
        sum(logsumexp(logits; dims=2))
end

function make_model(::Val{Symbol("science_irt-grsm_latent_reg_irt")}, data)
    W = Float64.(data["W"])
    adjustment = zeros(2, size(W, 2))
    adjustment[2, 1] = 1
    for k in 2:size(W, 2)
        adjustment[1, k] = mean(view(W, :, k))
        adjustment[2, k] = 2std(view(W, :, k))
    end
    W_adj = (W .- adjustment[1:1, :]) ./ adjustment[2:2, :]
    y = Int.(data["y"])
    max_category = fill(maximum(y), data["I"])
    levels = reshape(Float64.(0:maximum(y)), 1, :)
    response = y .+ 1
    response_indices = [
        n + (response[n] - 1) * data["N"] for n in 1:data["N"]
    ]
    return pdb_grsm_latent_reg_irt(
        data["I"], data["J"], data["N"], data["ii"], data["jj"], response,
        max_category, W_adj, levels, response_indices,
    )
end



# PosteriorDB Stan model: hier_2pl
@model function pdb_hier_2pl(I, J, N, ii, jj, y)
    theta ~ product_distribution(Fill(Flat(), J))
    xi1 ~ product_distribution(Fill(Flat(), I))
    xi2 ~ product_distribution(Fill(Flat(), I))
    mu ~ product_distribution(Fill(Flat(), 2))
    tau ~ product_distribution(Fill(FlatPos(0), 2))
    L_Omega ~ LKJCholesky(2, 4.0)
    xi = hcat(xi1, xi2)
    alpha = Base.materialize(Base.broadcasted(exp, xi1))
    beta = xi2
    L_Sigma = Diagonal(tau) * L_Omega.L
    Sigma = Distributions.PDMats.PDMat(Cholesky(Matrix(L_Sigma), 'L', 0))
    DynamicPPL.@addlogprob! loglikelihood(MvNormal(mu, Sigma), permutedims(xi))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(theta))), theta)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), mu[1])
    DynamicPPL.@addlogprob! logpdf(Exponential(inv(0.1)), tau[1])
    DynamicPPL.@addlogprob! logpdf(Normal(0, 5), mu[2])
    DynamicPPL.@addlogprob! logpdf(Exponential(inv(0.1)), tau[2])
    logits = alpha[ii] .* (theta[jj] .- beta[ii])
    DynamicPPL.@addlogprob! -sum(log1pexp.(y .* logits))
end

function make_model(::Val{Symbol("sat-hier_2pl")}, data)
    signs = 1 .- 2 .* Float64.(data["y"])
    return pdb_hier_2pl(
        data["I"], data["J"], data["N"], data["ii"], data["jj"], signs,
    )
end



# PosteriorDB Stan model: hierarchical_gp
@model function pdb_hierarchical_gp(N, N_states, N_regions, N_years_obs, N_years, state_region_ind, state_ind, region_ind, year_ind, y)
    # Keep homogeneous transforms structured, then materialize only at BLAS.
    GP_region_std ~ product_distribution(Fill(Flat(), N_years * N_regions))
    GP_state_std ~ product_distribution(Fill(Flat(), N_years * N_states))
    year_std ~ product_distribution(Fill(Flat(), N_years_obs))
    state_std ~ product_distribution(Fill(Flat(), N_states))
    region_std ~ product_distribution(Fill(Flat(), N_regions))
    tot_var ~ FlatPos(0)
    prop_var ~ Dirichlet(ones(17))
    mu ~ Flat()
    length_GP_region_long ~ FlatPos(0)
    length_GP_state_long ~ FlatPos(0)
    length_GP_region_short ~ FlatPos(0)
    length_GP_state_short ~ FlatPos(0)
    years = 1:N_years
    counts = fill(2, 17)
    vars = 17 * prop_var * tot_var
    sigma_year = sqrt(vars[1])
    sigma_region = sqrt(vars[2])
    sigma_state = Base.materialize(Base.broadcasted(sqrt, vars[3:end]))
    sigma_GP_region_long = sqrt(vars[13])
    sigma_GP_state_long = sqrt(vars[14])
    sigma_GP_region_short = sqrt(vars[15])
    sigma_GP_state_short = sqrt(vars[16])
    sigma_error_state_2 = sqrt(vars[17])
    region_re = sigma_region * region_std
    year_re = sigma_year * year_std
    state_re = sigma_state[state_region_ind] .* state_std
    cov_region = gp_exp_quad_cov(years, sigma_GP_region_long, length_GP_region_long) +
                 gp_exp_quad_cov(years, sigma_GP_region_short, length_GP_region_short)
    cov_state = gp_exp_quad_cov(years, sigma_GP_state_long, length_GP_state_long) +
                gp_exp_quad_cov(years, sigma_GP_state_short, length_GP_state_short)
    for year = 1:N_years
        cov_region[year, year] = cov_region[year, year] + 1.0e-6
        cov_state[year, year] = cov_state[year, year] + 1.0e-6
    end
    L_cov_region = cholesky(Symmetric(cov_region)).L
    L_cov_state = cholesky(Symmetric(cov_state)).L
    GP_region_matrix = Matrix(reshape(GP_region_std, N_years, N_regions))
    GP_state_matrix = Matrix(reshape(GP_state_std, N_years, N_states))
    GP_region = L_cov_region * GP_region_matrix
    GP_state = L_cov_state * GP_state_matrix
    obs_mu = zeros(typeof(mu + sigma_error_state_2), N)
    for n = 1:N
        obs_mu[n] = mu + year_re[year_ind[n]] + state_re[state_ind[n]] +
                    region_re[region_ind[n]] + GP_region[year_ind[n], region_ind[n]] +
                    GP_state[year_ind[n], state_ind[n]]
    end
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma_error_state_2), N)),
        y .- obs_mu,
    )
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(GP_region_std))), GP_region_std)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(GP_state_std))), GP_state_std)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(year_std))), year_std)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(state_std))), state_std)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(region_std))), region_std)
    DynamicPPL.@addlogprob! logpdf(Normal(0.5, 0.5), mu)
    DynamicPPL.@addlogprob! logpdf(Gamma(3, inv(3)), tot_var)
    DynamicPPL.@addlogprob! logpdf(Dirichlet(counts), prop_var)
    DynamicPPL.@addlogprob! logpdf(Weibull(30, 8), length_GP_region_long)
    DynamicPPL.@addlogprob! logpdf(Weibull(30, 8), length_GP_state_long)
    DynamicPPL.@addlogprob! logpdf(Weibull(30, 3), length_GP_region_short)
    DynamicPPL.@addlogprob! logpdf(Weibull(30, 3), length_GP_state_short)
end

function make_model(::Val{Symbol("state_wide_presidential_votes-hierarchical_gp")}, data)
    return pdb_hierarchical_gp(data["N"], data["N_states"], data["N_regions"], data["N_years_obs"], data["N_years"], data["state_region_ind"], data["state_ind"], data["region_ind"], data["year_ind"], data["y"])
end



# PosteriorDB Stan model: hmm_gaussian
@model function pdb_hmm_gaussian(T, K, y)
    pi1 ~ Dirichlet(ones(K))
    A ~ product_distribution(Fill(Dirichlet(ones(K)), K))
    mu ~ ordered(product_distribution(Fill(Flat(), K)))
    sigma ~ product_distribution(Fill(FlatPos(0), K))
    # Stan's first vectorized Normal density is a scalar sum which is then added
    # to every initial state. Preserve that model exactly.
    emissions = NormalEmissions(mu, sigma)
    # ProductDistribution stores transition simplex i in column i.
    y ~ HiddenMarkovModel(
        T, pi1, A, emissions, Val(:columns), Val(:summed_first),
    )
end

function make_model(::Val{Symbol("hmm_gaussian_simulated-hmm_gaussian")}, data)
    y = reshape(Float64.(data["y"]), :, 1)
    return pdb_hmm_gaussian(data["T"], data["K"], y)
end



# PosteriorDB Stan model: iohmm_reg
@model function pdb_iohmm_reg(T, K, M, y, u)
    pi1 ~ Dirichlet(ones(K))
    w ~ product_distribution(Fill(Normal(0, 5), K, M))
    b ~ product_distribution(Fill(Normal(0, 5), K, M))
    sigma ~ product_distribution(Fill(truncated(Normal(0, 3); lower=0), K))
            unA = (hcat(pi1, w * (u')[:, 2:end]))'
            A = copy(unA)
            for t = 2:T
                A[t, :] .= softmax(unA[t, :])
            end
            logA = log.(A)
            emission_mu = u * b'
            emissions = NormalEmissions(emission_mu, sigma)
            y ~ HiddenMarkovModel(
                T, pi1, logA, emissions, Val(:source_logprob),
            )
end

function make_model(::Val{Symbol("iohmm_reg_simulated-iohmm_reg")}, data)
    y = reshape(Float64.(data["y"]), :, 1)
    return pdb_iohmm_reg(
        data["T"], data["K"], data["M"], y, Float64.(data["u"]),
    )
end



# PosteriorDB Stan model: irt_2pl
@model function pdb_irt_2pl(I, J, y)
    sigma_theta ~ FlatPos(0)
    theta ~ product_distribution(Fill(Flat(), J))
    sigma_a ~ FlatPos(0)
    a ~ product_distribution(Fill(FlatPos(0), I))
    mu_b ~ Flat()
    sigma_b ~ FlatPos(0)
    b ~ product_distribution(Fill(Flat(), I))
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2), sigma_theta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, sigma_theta), size(theta))), theta)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2), sigma_a)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(LogNormal(0, sigma_a), size(a))), a)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 5), mu_b)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2), sigma_b)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_b, sigma_b), size(b))), b)
    for i = 1:I
        logits = a[i] .* (theta .- b[i])
        DynamicPPL.@addlogprob! logpdf(
            product_distribution(map(BernoulliLogit, logits)), y[i, :],
        )
    end
end

function make_model(::Val{Symbol("irt_2pl-irt_2pl")}, data)
    return pdb_irt_2pl(data["I"], data["J"], data["y"])
end



# PosteriorDB Stan model: kronecker_gp
@model function pdb_kronecker_gp(n1, n2, xd, y)
    var1 ~ LogNormal(0, 1)
    bw1 ~ truncated(Cauchy(0, 2.5); lower=0)
    L ~ LKJCholesky(n2, 2.0)
    sigma1 ~ truncated(LogNormal(0, 1); lower=1.0e-5)
    Lm = Matrix(L.L)
    y ~ RBFKroneckerNormal(var1, bw1, Lm, sigma1, xd)
end

function make_model(::Val{Symbol("synthetic_grid_RBF_kernels-kronecker_gp")}, data)
    x = Float64.(data["x1"])
    xd = -((x .- x') .^ 2)
    return pdb_kronecker_gp(
        data["n1"], data["n2"], xd, Matrix{Float64}(data["y"]),
    )
end



# PosteriorDB Stan model: ldaK2
@model function pdb_ldaK2(V, M, N, w, doc)
    K = 2
    theta ~ product_distribution(Fill(Dirichlet(ones(K)), M))
    phi ~ product_distribution(Fill(Dirichlet(ones(V)), K))
    logits = permutedims(log.(view(theta, :, doc))) .+
        log.(view(phi, w, :))
    DynamicPPL.@addlogprob! sum(logsumexp(logits; dims=2))
end

function make_model(::Val{Symbol("three_docs1200-ldaK2")}, data)
    return pdb_ldaK2(data["V"], data["M"], data["N"], data["w"], data["doc"])
end

function make_model(::Val{Symbol("three_men1-ldaK2")}, data)
    return pdb_ldaK2(data["V"], data["M"], data["N"], data["w"], data["doc"])
end

function make_model(::Val{Symbol("three_men2-ldaK2")}, data)
    return pdb_ldaK2(data["V"], data["M"], data["N"], data["w"], data["doc"])
end

function make_model(::Val{Symbol("three_men3-ldaK2")}, data)
    return pdb_ldaK2(data["V"], data["M"], data["N"], data["w"], data["doc"])
end



# PosteriorDB Stan model: ldaK5
@model function pdb_ldaK5(V, M, N, w, doc, alpha, beta)
    theta ~ product_distribution(Fill(Dirichlet(alpha), M))
    phi ~ product_distribution(Fill(Dirichlet(beta), 5))
    logits = permutedims(log.(view(theta, :, doc))) .+
        log.(view(phi, w, :))
    DynamicPPL.@addlogprob! sum(logsumexp(logits; dims=2))
end

function make_model(::Val{Symbol("prideprejudice_chapter-ldaK5")}, data)
    return pdb_ldaK5(data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])
end

function make_model(::Val{Symbol("prideprejudice_paragraph-ldaK5")}, data)
    return pdb_ldaK5(data["V"], data["M"], data["N"], data["w"], data["doc"], data["alpha"], data["beta"])
end



# PosteriorDB Stan model: logistic_regression_rhs
@model function pdb_logistic_regression_rhs(n, d, y, x, scale_icept, scale_global, nu_global, nu_local, slab_scale, slab_df)
    x = x.value
    beta0 ~ Normal(0.0, scale_icept)
    z ~ product_distribution(Fill(Normal(), d))
    tau ~ truncated(
        LocationScale(0.0, 2.0scale_global, TDist(nu_global)); lower=0.0,
    )
    lambda ~ product_distribution(Fill(
        truncated(LocationScale(0.0, 1.0, TDist(nu_local)); lower=0.0), d,
    ))
    caux ~ InverseGamma(0.5slab_df, 0.5slab_df)
    c2 = slab_scale^2 * caux
    tau2 = tau^2
    beta = tau .* z .* lambda .* sqrt.(c2 ./ (c2 .+ tau2 .* abs2.(lambda)))
    logits = beta0 .+ x * beta
    DynamicPPL.@addlogprob! dot(y, logits) - sum(log1pexp.(logits))
end

function make_model(::Val{Symbol("ovarian-logistic_regression_rhs")}, data)
    X = _FixedData(Matrix{Float64}(data["x"]))
    return pdb_logistic_regression_rhs(data["n"], data["d"], data["y"], X, data["scale_icept"], data["scale_global"], data["nu_global"], data["nu_local"], data["slab_scale"], data["slab_df"])
end

function make_model(::Val{Symbol("prostate-logistic_regression_rhs")}, data)
    X = _FixedData(Matrix{Float64}(data["x"]))
    return pdb_logistic_regression_rhs(data["n"], data["d"], data["y"], X, data["scale_icept"], data["scale_global"], data["nu_global"], data["nu_local"], data["slab_scale"], data["slab_df"])
end



# PosteriorDB Stan model: losscurve_sislob
@model function pdb_losscurve_sislob(growthmodel_id, n_data, n_time, n_cohort, cohort_id, t_idx, cohort_maxtime, t_value, premium, loss)
    omega ~ FlatPos(0)
    theta ~ FlatPos(0)
    LR ~ product_distribution(Fill(FlatPos(0), n_cohort))
    mu_LR ~ Flat()
    sd_LR ~ FlatPos(0)
    loss_sd ~ FlatPos(0)
    gf = if growthmodel_id == 1
        1 .- exp.(-((t_value ./ theta) .^ omega))
    else
        powered_time = t_value .^ omega
        powered_time ./ (powered_time .+ theta^omega)
    end
            DynamicPPL.@addlogprob! logpdf(Normal(0, 0.5), mu_LR)
            DynamicPPL.@addlogprob! logpdf(LogNormal(0, 0.5), sd_LR)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(LogNormal(mu_LR, sd_LR), size(LR))), LR)
            DynamicPPL.@addlogprob! logpdf(LogNormal(0, 0.7), loss_sd)
            DynamicPPL.@addlogprob! logpdf(LogNormal(0, 0.5), omega)
            DynamicPPL.@addlogprob! logpdf(LogNormal(0, 0.5), theta)
            scale = (loss_sd .* premium)[cohort_id]
            location = LR[cohort_id] .* premium[cohort_id] .* gf[t_idx]
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Normal(), size(loss))),
                (loss .- location) ./ scale,
            ) - sum(log, scale)
end

function make_model(::Val{Symbol("loss_curves-losscurve_sislob")}, data)
    return pdb_losscurve_sislob(data["growthmodel_id"], data["n_data"], data["n_time"], data["n_cohort"], data["cohort_id"], data["t_idx"], data["cohort_maxtime"], data["t_value"], data["premium"], data["loss"])
end



# PosteriorDB Stan model: low_dim_gauss_mix_collapse
@model function pdb_low_dim_gauss_mix_collapse(N, y)
    mu ~ product_distribution(Fill(Flat(), 2))
    sigma ~ product_distribution(Fill(FlatPos(0), 2))
    theta ~ Uniform(0, 1)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 2), size(sigma))), sigma)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 2), size(mu))), mu)
    DynamicPPL.@addlogprob! logpdf(Beta(5, 5), theta)
    for n = 1:N
        DynamicPPL.@addlogprob! logaddexp(
            log(theta) + logpdf(Normal(mu[1], sigma[1]), y[n]),
            log1p(-theta) + logpdf(Normal(mu[2], sigma[2]), y[n]),
        )
    end
end

function make_model(::Val{Symbol("low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse")}, data)
    return pdb_low_dim_gauss_mix_collapse(data["N"], data["y"])
end



# PosteriorDB Stan model: lsat_model
@model function pdb_lsat_model(N, T, r)
    alpha ~ product_distribution(Fill(Flat(), T))
    theta ~ product_distribution(Fill(Flat(), N))
    beta ~ FlatPos(0)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 100.0), size(alpha))), alpha)
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(theta))), theta)
            DynamicPPL.@addlogprob! logpdf(Normal(0.0, 100.0), beta)
            for k = 1:T
                logits = beta .* theta .- alpha[k]
                DynamicPPL.@addlogprob! logpdf(
                    product_distribution(map(BernoulliLogit, logits)), r[k, :],
                )
            end
end

function make_model(::Val{Symbol("lsat_data-lsat_model")}, data)
    N, R, T = data["N"], data["R"], data["T"]
    culm, response = data["culm"], data["response"]
    r = zeros(Int, T, N)
    for j in 1:culm[1], k in 1:T
        r[k, j] = response[1, k]
    end
    for i in 2:R, j in (culm[i - 1] + 1):culm[i], k in 1:T
        r[k, j] = response[i, k]
    end
    return pdb_lsat_model(N, T, r)
end



# PosteriorDB Stan model: multi_occupancy
@model function pdb_multi_occupancy(J, K, n, X, S)
    alpha ~ Flat()
    beta ~ Flat()
    Omega ~ Uniform(0, 1)
    rho_uv ~ Uniform(-1, 1)
    sigma_uv ~ product_distribution(Fill(FlatPos(0), 2))
    uv1 ~ product_distribution(Fill(Flat(), S))
    uv2 ~ product_distribution(Fill(Flat(), S))
    uv = hcat(uv1, uv2)
    logit_psi = Base.materialize(Base.broadcasted(+, uv1, alpha))
    logit_theta = Base.materialize(Base.broadcasted(+, uv2, beta))
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2.5), alpha)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2.5), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Cauchy(0, 2.5), size(sigma_uv))), sigma_uv)
    DynamicPPL.@addlogprob! logpdf(Beta(2, 2), (rho_uv + 1) / 2)
    covariance12 = sigma_uv[1] * sigma_uv[2] * rho_uv
    covariance = [
        sigma_uv[1]^2 covariance12
        covariance12 sigma_uv[2]^2
    ]
    uv_distribution = MvNormal(
        zeros(2), Symmetric(covariance),
    )
    DynamicPPL.@addlogprob! loglikelihood(uv_distribution, uv')
    DynamicPPL.@addlogprob! logpdf(Beta(2, 2), Omega)
    for i = 1:n
        DynamicPPL.@addlogprob! logpdf(Bernoulli(Omega), 1)
        for j = 1:J
            occupied = logpdf(BernoulliLogit(logit_psi[i]), 1) +
                logpdf(BinomialLogit(K, logit_theta[i]), X[i, j])
            if X[i, j] > 0
                DynamicPPL.@addlogprob! occupied
            else
                DynamicPPL.@addlogprob! logaddexp(
                    occupied, logpdf(BernoulliLogit(logit_psi[i]), 0),
                )
            end
        end
    end
    for i = n + 1:S
        unobserved = logaddexp(
            logpdf(BernoulliLogit(logit_psi[i]), 1) +
                logpdf(BinomialLogit(K, logit_theta[i]), 0),
            logpdf(BernoulliLogit(logit_psi[i]), 0),
        )
        DynamicPPL.@addlogprob! logaddexp(
            logpdf(Bernoulli(Omega), 0),
            logpdf(Bernoulli(Omega), 1) + J * unobserved,
        )
    end
end

function make_model(::Val{Symbol("butterfly-multi_occupancy")}, data)
    return pdb_multi_occupancy(data["J"], data["K"], data["n"], data["X"], data["S"])
end



# PosteriorDB Stan model: nes_logit_model
@model function pdb_nes_logit_model(X, vote)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), size(X, 2)))
    logits = alpha .+ X * beta
    vote ~ product_distribution(map(BernoulliLogit, logits))
end

function make_model(::Val{Symbol("nes_logit_data-nes_logit_model")}, data)
    X = reshape(Float64.(data["income"]), data["N"], 1)
    return pdb_nes_logit_model(_FixedData(X), data["vote"])
end



# PosteriorDB Stan model: nn_rbm1bJ10
@model function pdb_nn_rbm1bJ10(N, M, x, K, y)
    x = x.value
    J = 10
    nu_alpha = 0.5
    s2_0_alpha = (0.05 / M ^ (1 / nu_alpha)) ^ 2
    nu_beta = 0.5
    s2_0_beta = (0.05 / J ^ (1 / nu_beta)) ^ 2
    sigma2_alpha ~ InverseGamma(nu_alpha / 2, (nu_alpha * s2_0_alpha) / 2)
    sigma2_beta ~ InverseGamma(nu_beta / 2, (nu_beta * s2_0_beta) / 2)
    alpha ~ product_distribution(Fill(Normal(0, sqrt(sigma2_alpha)), M * J))
    beta ~ product_distribution(Fill(Normal(0, sqrt(sigma2_beta)), J * (K - 1)))
    alpha1 ~ product_distribution(Fill(Normal(0, 1), J))
    beta1 ~ product_distribution(Fill(Normal(0, 1), K - 1))
    hidden = tanh.(x * reshape(alpha, M, J) .+ alpha1')
    logits = hidden * reshape(beta, J, K - 1) .+ beta1'
    y ~ product_distribution(map(axes(logits, 1)) do i
        Categorical(softmax(vcat(1.0, collect(view(logits, i, :)))))
    end)
end

function make_model(::Val{Symbol("mnist_100-nn_rbm1bJ10")}, data)
    return pdb_nn_rbm1bJ10(
        data["N"], data["M"], _FixedData(Float64.(data["x"])), data["K"], data["y"],
    )
end



# PosteriorDB Stan model: nn_rbm1bJ100
@model function pdb_nn_rbm1bJ100(N, M, x, K, y)
    x = x.value
    J = 100
    nu_alpha = 0.5
    s2_0_alpha = (0.05 / M ^ (1 / nu_alpha)) ^ 2
    nu_beta = 0.5
    s2_0_beta = (0.05 / J ^ (1 / nu_beta)) ^ 2
    sigma2_alpha ~ InverseGamma(nu_alpha / 2, (nu_alpha * s2_0_alpha) / 2)
    sigma2_beta ~ InverseGamma(nu_beta / 2, (nu_beta * s2_0_beta) / 2)
    alpha ~ product_distribution(Fill(Normal(0, sqrt(sigma2_alpha)), M * J))
    beta ~ product_distribution(Fill(Normal(0, sqrt(sigma2_beta)), J * (K - 1)))
    alpha1 ~ product_distribution(Fill(Normal(0, 1), J))
    beta1 ~ product_distribution(Fill(Normal(0, 1), K - 1))
    hidden = tanh.(x * reshape(alpha, M, J) .+ alpha1')
    logits = hidden * reshape(beta, J, K - 1) .+ beta1'
    y ~ product_distribution(map(axes(logits, 1)) do i
        Categorical(softmax(vcat(1.0, collect(view(logits, i, :)))))
    end)
end

function make_model(::Val{Symbol("mnist-nn_rbm1bJ100")}, data)
    return pdb_nn_rbm1bJ100(
        data["N"], data["M"], _FixedData(Float64.(data["x"])), data["K"], data["y"],
    )
end



# PosteriorDB Stan model: normal_mixture
@model function pdb_normal_mixture(N, y)
    theta ~ Uniform(0, 1)
    mu ~ product_distribution(Fill(Flat(), 2))
    for k = 1:2
        DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu[k])
    end
    for n = 1:N
        DynamicPPL.@addlogprob! logaddexp(
            log(theta) + logpdf(Normal(mu[1], 1.0), y[n]),
            log1p(-theta) + logpdf(Normal(mu[2], 1.0), y[n]),
        )
    end
end

function make_model(::Val{Symbol("normal_2-normal_mixture")}, data)
    return pdb_normal_mixture(data["N"], data["y"])
end



# PosteriorDB Stan model: normal_mixture_k
@model function pdb_normal_mixture_k(K, N, y)
    theta ~ Dirichlet(ones(K))
    mu ~ product_distribution(Fill(Flat(), K))
    sigma ~ product_distribution(Fill(Uniform(0.0, 10.0), K))
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, 10.0), size(mu))), mu)
    for n = 1:N
        ps = map(
            (weight, location, scale) ->
                log(weight) + logpdf(Normal(location, scale), y[n]),
            theta, mu, sigma,
        )
        DynamicPPL.@addlogprob! logsumexp(ps)
    end
end

function make_model(::Val{Symbol("normal_5-normal_mixture_k")}, data)
    return pdb_normal_mixture_k(data["K"], data["N"], data["y"])
end



# PosteriorDB Stan model: pilots
@model function pdb_pilots(N, n_groups, n_scenarios, group_id, scenario_id, y)
    a ~ product_distribution(Fill(Flat(), n_groups))
    b ~ product_distribution(Fill(Flat(), n_scenarios))
    mu_a ~ Flat()
    mu_b ~ Flat()
    sigma_a ~ Uniform(0, 100)
    sigma_b ~ Uniform(0, 100)
    sigma_y ~ Uniform(0, 100)
    y_hat = Base.materialize(Base.broadcasted(+, a[group_id], b[scenario_id]))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), mu_a)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(10mu_a, sigma_a), size(a))), a)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), mu_b)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(10mu_b, sigma_b), size(b))), b)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma_y), N)), y .- y_hat,
    )
end

function make_model(::Val{Symbol("pilots-pilots")}, data)
    return pdb_pilots(data["N"], data["n_groups"], data["n_scenarios"], data["group_id"], data["scenario_id"], data["y"])
end



# PosteriorDB Stan model: prophet
@model function pdb_prophet(T, K, t, cap, y, S, t_change, A, A_tchange, X_a, X_m, sigmas, tau, trend_indicator)
    k ~ Flat()
    m ~ Flat()
    delta ~ product_distribution(Fill(Flat(), S))
    sigma_obs ~ FlatPos(0)
    beta ~ product_distribution(Fill(Flat(), K))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 5), k)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 5), m)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Laplace(0, tau), size(delta))), delta,
    )
    DynamicPPL.@addlogprob! logpdf(Normal(0, 0.5), sigma_obs)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(), size(beta))), beta ./ sigmas,
    ) - sum(log, sigmas)
    if trend_indicator == 0
        trend = (k .+ A * delta) .* t .+ (m .- A_tchange * delta)
        prediction = trend .* (1 .+ X_m * beta) + X_a * beta
        DynamicPPL.@addlogprob! logpdf(
            product_distribution(Fill(Normal(0, sigma_obs), T)),
            y .- prediction,
        )
    elseif trend_indicator == 1
        gamma = zeros(S)
        segment_rate = vcat(k, k + cumsum(delta))
        shifted_offset = m
        for i = 1:S
            gamma[i] = (t_change[i] - shifted_offset) *
                (1 - segment_rate[i] / segment_rate[i + 1])
            shifted_offset += gamma[i]
        end
        trend = cap .* logistic.(
            (k .+ A * delta) .* ((t .- m) .- A * gamma),
        )
        prediction = trend .* (1 .+ X_m * beta) + X_a * beta
        DynamicPPL.@addlogprob! logpdf(
            product_distribution(Fill(Normal(0, sigma_obs), T)),
            y .- prediction,
        )
    end
end

function make_model(::Val{Symbol("rstan_downloads-prophet")}, data)
    T = data["T"]
    S = data["S"]
    t = Float64.(data["t"])
    t_change = Float64.(data["t_change"])
    A = zeros(T, S)
    active = zeros(S)
    changepoint = 1
    for i in 1:T
        while changepoint <= S && t[i] >= t_change[changepoint]
            active[changepoint] = 1.0
            changepoint += 1
        end
        A[i, :] .= active
    end
    A_tchange = A .* reshape(t_change, 1, S)
    X = Float64.(data["X"])
    X_a = X .* reshape(Float64.(data["s_a"]), 1, :)
    X_m = X .* reshape(Float64.(data["s_m"]), 1, :)
    return pdb_prophet(
        T, data["K"], t, Float64.(data["cap"]), Float64.(data["y"]),
        S, t_change, A, A_tchange, X_a, X_m, Float64.(data["sigmas"]),
        data["tau"], data["trend_indicator"],
    )
end



# PosteriorDB Stan model: radon_county
@model function pdb_radon_county(J, county_design, y)
    a ~ product_distribution(Fill(Flat(), J))
    mu_a ~ Flat()
    sigma_a ~ Uniform(0.0, 100.0)
    sigma_y ~ Uniform(0.0, 100.0)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), mu_a)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_a, sigma_a), size(a))), a)
    prediction = county_design.value * a
    y ~ MvNormal(prediction, sigma_y)
end

function make_model(::Val{Symbol("radon_mod-radon_county")}, data)
    N = data["N"]
    J = data["J"]
    county_design = sparse(
        1:N, Int.(data["county"]), ones(Float64, N), N, J,
    )
    return pdb_radon_county(
        J, _FixedData(county_design), Float64.(data["y"]),
    )
end



# PosteriorDB Stan model: radon_county_intercept
@model function pdb_radon_county_intercept(N, J, county_idx, floor_measure, log_radon)
    alpha ~ product_distribution(Fill(Flat(), J))
    beta ~ Flat()
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 10), size(alpha))), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), beta)
    for n = 1:N
        mu = alpha[county_idx[n]] + beta * floor_measure[n]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_county_intercept")}, data)
    return pdb_radon_county_intercept(data["N"], data["J"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_county_intercept")}, data)
    return pdb_radon_county_intercept(data["N"], data["J"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_hierarchical_intercept_centered
@model function pdb_radon_hierarchical_intercept_centered(J, N, county_idx, log_uppm, floor_measure, log_radon)
    alpha ~ product_distribution(Fill(Flat(), J))
    beta ~ product_distribution(Fill(Flat(), 2))
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 10), size(beta))), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sigma_alpha), size(alpha))), alpha)
    for n = 1:N
        muj = alpha[county_idx[n]] + log_uppm[n] * beta[1]
        mu = muj + floor_measure[n] * beta[2]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_hierarchical_intercept_centered")}, data)
    return pdb_radon_hierarchical_intercept_centered(data["J"], data["N"], data["county_idx"], data["log_uppm"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_hierarchical_intercept_centered")}, data)
    return pdb_radon_hierarchical_intercept_centered(data["J"], data["N"], data["county_idx"], data["log_uppm"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_hierarchical_intercept_noncentered
@model function pdb_radon_hierarchical_intercept_noncentered(J, N, county_idx, log_uppm, floor_measure, log_radon)
    alpha_raw ~ product_distribution(Fill(Flat(), J))
    beta ~ product_distribution(Fill(Flat(), 2))
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    alpha = Base.materialize(Base.broadcasted(+, mu_alpha, Base.broadcasted(*, sigma_alpha, alpha_raw)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 10), size(beta))), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(alpha_raw))), alpha_raw)
    for n = 1:N
        muj = alpha[county_idx[n]] + log_uppm[n] * beta[1]
        mu = muj + floor_measure[n] * beta[2]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_hierarchical_intercept_noncentered")}, data)
    return pdb_radon_hierarchical_intercept_noncentered(data["J"], data["N"], data["county_idx"], data["log_uppm"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_hierarchical_intercept_noncentered")}, data)
    return pdb_radon_hierarchical_intercept_noncentered(data["J"], data["N"], data["county_idx"], data["log_uppm"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_partially_pooled_centered
@model function pdb_radon_partially_pooled_centered(N, J, county_idx, log_radon)
    alpha ~ product_distribution(Fill(Flat(), J))
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sigma_alpha), size(alpha))), alpha)
    for n = 1:N
        mu = alpha[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_partially_pooled_centered")}, data)
    return pdb_radon_partially_pooled_centered(data["N"], data["J"], data["county_idx"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_partially_pooled_centered")}, data)
    return pdb_radon_partially_pooled_centered(data["N"], data["J"], data["county_idx"], data["log_radon"])
end



# PosteriorDB Stan model: radon_partially_pooled_noncentered
@model function pdb_radon_partially_pooled_noncentered(N, J, county_idx, log_radon)
    alpha_raw ~ product_distribution(Fill(Flat(), J))
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    alpha = Base.materialize(Base.broadcasted(+, mu_alpha, Base.broadcasted(*, sigma_alpha, alpha_raw)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(alpha_raw))), alpha_raw)
    for n = 1:N
        mu = alpha[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_partially_pooled_noncentered")}, data)
    return pdb_radon_partially_pooled_noncentered(data["N"], data["J"], data["county_idx"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_partially_pooled_noncentered")}, data)
    return pdb_radon_partially_pooled_noncentered(data["N"], data["J"], data["county_idx"], data["log_radon"])
end



# PosteriorDB Stan model: radon_pooled
@model function pdb_radon_pooled(N, floor_measure, log_radon)
    alpha ~ Flat()
    beta ~ Flat()
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), beta)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma_y), size(log_radon))),
        log_radon .- (alpha .+ beta .* floor_measure),
    )
end

function make_model(::Val{Symbol("radon_all-radon_pooled")}, data)
    return pdb_radon_pooled(data["N"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_pooled")}, data)
    return pdb_radon_pooled(data["N"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_intercept_centered
@model function pdb_radon_variable_intercept_centered(J, N, county_idx, floor_measure, log_radon)
    alpha ~ product_distribution(Fill(Flat(), J))
    beta ~ Flat()
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sigma_alpha), size(alpha))), alpha)
    for n = 1:N
        mu = alpha[county_idx[n]] + floor_measure[n] * beta
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_intercept_centered")}, data)
    return pdb_radon_variable_intercept_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_intercept_centered")}, data)
    return pdb_radon_variable_intercept_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_intercept_noncentered
@model function pdb_radon_variable_intercept_noncentered(J, N, county_idx, floor_measure, log_radon)
    alpha_raw ~ product_distribution(Fill(Flat(), J))
    beta ~ Flat()
    mu_alpha ~ Flat()
    sigma_alpha ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    alpha = Base.materialize(Base.broadcasted(+, mu_alpha, Base.broadcasted(*, sigma_alpha, alpha_raw)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(alpha_raw))), alpha_raw)
    for n = 1:N
        mu = alpha[county_idx[n]] + floor_measure[n] * beta
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_intercept_noncentered")}, data)
    return pdb_radon_variable_intercept_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_intercept_noncentered")}, data)
    return pdb_radon_variable_intercept_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_intercept_slope_centered
@model function pdb_radon_variable_intercept_slope_centered(J, N, county_idx, floor_measure, log_radon)
    sigma_y ~ FlatPos(0)
    sigma_alpha ~ FlatPos(0)
    sigma_beta ~ FlatPos(0)
    alpha ~ product_distribution(Fill(Flat(), J))
    beta ~ product_distribution(Fill(Flat(), J))
    mu_alpha ~ Flat()
    mu_beta ~ Flat()
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_beta)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sigma_alpha), size(alpha))), alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_beta, sigma_beta), size(beta))), beta)
    for n = 1:N
        mu = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_intercept_slope_centered")}, data)
    return pdb_radon_variable_intercept_slope_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_intercept_slope_centered")}, data)
    return pdb_radon_variable_intercept_slope_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_intercept_slope_noncentered
@model function pdb_radon_variable_intercept_slope_noncentered(J, N, county_idx, floor_measure, log_radon)
    sigma_y ~ FlatPos(0)
    sigma_alpha ~ FlatPos(0)
    sigma_beta ~ FlatPos(0)
    alpha_raw ~ product_distribution(Fill(Flat(), J))
    beta_raw ~ product_distribution(Fill(Flat(), J))
    mu_alpha ~ Flat()
    mu_beta ~ Flat()
    alpha = Base.materialize(Base.broadcasted(+, mu_alpha, Base.broadcasted(*, sigma_alpha, alpha_raw)))
    beta = Base.materialize(Base.broadcasted(+, mu_beta, Base.broadcasted(*, sigma_beta, beta_raw)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_beta)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(alpha_raw))), alpha_raw)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(beta_raw))), beta_raw)
    for n = 1:N
        mu = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_intercept_slope_noncentered")}, data)
    return pdb_radon_variable_intercept_slope_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_intercept_slope_noncentered")}, data)
    return pdb_radon_variable_intercept_slope_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_slope_centered
@model function pdb_radon_variable_slope_centered(J, N, county_idx, floor_measure, log_radon)
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), J))
    mu_beta ~ Flat()
    sigma_beta ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_beta)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_beta, sigma_beta), size(beta))), beta)
    for n = 1:N
        mu = alpha + floor_measure[n] * beta[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_slope_centered")}, data)
    return pdb_radon_variable_slope_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_slope_centered")}, data)
    return pdb_radon_variable_slope_centered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: radon_variable_slope_noncentered
@model function pdb_radon_variable_slope_noncentered(J, N, county_idx, floor_measure, log_radon)
    alpha ~ Flat()
    beta_raw ~ product_distribution(Fill(Flat(), J))
    mu_beta ~ Flat()
    sigma_beta ~ FlatPos(0)
    sigma_y ~ FlatPos(0)
    beta = Base.materialize(Base.broadcasted(+, mu_beta, Base.broadcasted(*, sigma_beta, beta_raw)))
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_y)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), sigma_beta)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 10), mu_beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0, 1), size(beta_raw))), beta_raw)
    for n = 1:N
        mu = alpha + floor_measure[n] * beta[county_idx[n]]
        DynamicPPL.@addlogprob! logpdf(Normal(mu, sigma_y), log_radon[n])
    end
end

function make_model(::Val{Symbol("radon_all-radon_variable_slope_noncentered")}, data)
    return pdb_radon_variable_slope_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end

function make_model(::Val{Symbol("radon_mn-radon_variable_slope_noncentered")}, data)
    return pdb_radon_variable_slope_noncentered(data["J"], data["N"], data["county_idx"], data["floor_measure"], data["log_radon"])
end



# PosteriorDB Stan model: rats_model
@model function pdb_rats_model(N, Npts, rat, x, y, xbar)
    x_ = x
    alpha ~ product_distribution(Fill(Flat(), N))
    beta ~ product_distribution(Fill(Flat(), N))
    mu_alpha ~ Flat()
    mu_beta ~ Flat()
    sigma_y ~ FlatPos(0)
    sigma_alpha ~ FlatPos(0)
    sigma_beta ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 100), mu_alpha)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 100), mu_beta)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_alpha, sigma_alpha), size(alpha))), alpha)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu_beta, sigma_beta), size(beta))), beta)
    for n = 1:Npts
        irat = rat[n]
        DynamicPPL.@addlogprob! logpdf(Normal(alpha[irat] + beta[irat] * (x_[n] - xbar), sigma_y), y[n])
    end
end

function make_model(::Val{Symbol("rats_data-rats_model")}, data)
    return pdb_rats_model(data["N"], data["Npts"], data["rat"], data["x"], data["y"], data["xbar"])
end



# PosteriorDB Stan model: seeds_centered_model
@model function pdb_seeds_centered_model(I, n, N, x1, x2, x1x2)
    alpha0 ~ Flat()
    alpha1 ~ Flat()
    alpha12 ~ Flat()
    alpha2 ~ Flat()
    c ~ product_distribution(Fill(Flat(), I))
    sigma ~ FlatPos(0)
    b = Base.materialize(Base.broadcasted(-, c, mean(c)))
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha0)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha1)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha2)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha12)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 1), sigma)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, sigma), size(c))), c)
    logits = alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(BinomialLogit, N, logits)), n,
    )
end

function make_model(::Val{Symbol("seeds_data-seeds_centered_model")}, data)
    x1, x2 = data["x1"], data["x2"]
    return pdb_seeds_centered_model(
        data["I"], data["n"], data["N"], x1, x2, x1 .* x2,
    )
end



# PosteriorDB Stan model: seeds_model
@model function pdb_seeds_model(I, n, N, x1, x2, x1x2)
    alpha0 ~ Flat()
    alpha1 ~ Flat()
    alpha12 ~ Flat()
    alpha2 ~ Flat()
    tau ~ FlatPos(0)
    b ~ product_distribution(Fill(Flat(), I))
    sigma = 1.0 / sqrt(tau)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), alpha0)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), alpha1)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), alpha2)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), alpha12)
    DynamicPPL.@addlogprob! logpdf(Gamma(0.001, inv(0.001)), tau)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, sigma), size(b))), b)
    logits = alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(BinomialLogit, N, logits)), n,
    )
end

function make_model(::Val{Symbol("seeds_data-seeds_model")}, data)
    x1, x2 = data["x1"], data["x2"]
    return pdb_seeds_model(
        data["I"], data["n"], data["N"], x1, x2, x1 .* x2,
    )
end



# PosteriorDB Stan model: seeds_stanified_model
@model function pdb_seeds_stanified_model(I, n, N, x1, x2, x1x2)
    alpha0 ~ Flat()
    alpha1 ~ Flat()
    alpha12 ~ Flat()
    alpha2 ~ Flat()
    b ~ product_distribution(Fill(Flat(), I))
    sigma ~ FlatPos(0)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha0)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha1)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha2)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1.0), alpha12)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 1), sigma)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(0.0, sigma), size(b))), b)
    logits = alpha0 .+ alpha1 .* x1 .+ alpha2 .* x2 .+ alpha12 .* x1x2 .+ b
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(BinomialLogit, N, logits)), n,
    )
end

function make_model(::Val{Symbol("seeds_data-seeds_stanified_model")}, data)
    x1, x2 = data["x1"], data["x2"]
    return pdb_seeds_stanified_model(
        data["I"], data["n"], data["N"], x1, x2, x1 .* x2,
    )
end



# PosteriorDB Stan model: sesame_one_pred_a
@model function pdb_sesame_one_pred_a(N, encouraged, watched)
    beta ~ product_distribution(Fill(Flat(), 2))
    sigma ~ FlatPos(0.0)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(watched))),
        watched .- (beta[1] .+ beta[2] .* encouraged),
    )
end

function make_model(::Val{Symbol("sesame_data-sesame_one_pred_a")}, data)
    return pdb_sesame_one_pred_a(data["N"], data["encouraged"], data["watched"])
end



# PosteriorDB Stan model: sir
@model function pdb_sir(N_t, t0, t, y0, stoi_hat, B_hat)
    kappa = 1.0e6
    beta ~ FlatPos(0)
    gamma ~ FlatPos(0)
    xi ~ FlatPos(0)
    delta ~ FlatPos(0)
            sir_rhs! = function (derivative, state, theta, _)
                derivative[1] = -theta[1] * state[4] / (state[4] + theta[2]) * state[1]
                derivative[2] = theta[1] * state[4] / (state[4] + theta[2]) * state[1] -
                    theta[3] * state[2]
                derivative[3] = theta[3] * state[2]
                derivative[4] = theta[4] * state[2] - theta[5] * state[4]
                return nothing
            end
            y = _solve_ode(
                DP5(), sir_rhs!, y0, t0, t,
                [beta, kappa, gamma, xi, delta], 1e-6, 1e-6, 100_000_000,
            )
            DynamicPPL.@addlogprob! logpdf(Cauchy(0, 2.5), beta)
            DynamicPPL.@addlogprob! logpdf(Cauchy(0, 1), gamma)
            DynamicPPL.@addlogprob! logpdf(Cauchy(0, 25), xi)
            DynamicPPL.@addlogprob! logpdf(Cauchy(0, 1), delta)
            DynamicPPL.@addlogprob! logpdf(Poisson(y0[1] - y[1, 1]), stoi_hat[1])
            for n = 2:N_t
                DynamicPPL.@addlogprob! logpdf(Poisson(max(1.0e-16, y[n - 1, 1] - y[n, 1])), stoi_hat[n])
            end
            if any(y[:, 4] .<= 0)
                DynamicPPL.@addlogprob! -Inf
                return
            end
            DynamicPPL.@addlogprob! logpdf(product_distribution(map(
                m -> LogNormal(m, 0.15), log.(y[:, 4]),
            )), B_hat)
end

function make_model(::Val{Symbol("sir-sir")}, data)
    return pdb_sir(
        data["N_t"], _FixedData(0.0), _FixedData(Float64.(data["t"])),
        Float64.(data["y0"]),
        data["stoi_hat"], data["B_hat"],
    )
end



# PosteriorDB Stan model: soil_incubation
@model function pdb_soil_incubation(totalC_t0, t0, N_t, ts, eCO2mean)
    k1 ~ FlatPos(0)
    k2 ~ FlatPos(0)
    alpha21 ~ FlatPos(0)
    alpha12 ~ FlatPos(0)
    gamma ~ Uniform(0, 1)
    sigma ~ FlatPos(0)
    carbon1_t0 = gamma * totalC_t0
    carbon2_t0 = (1 - gamma) * totalC_t0
    half_trace = -(k1 + k2) / 2
    half_difference = (k2 - k1) / 2
    eigen_gap = sqrt(half_difference^2 + alpha12 * alpha21 * k1 * k2)
    transformed1 = half_difference * carbon1_t0 + alpha12 * k2 * carbon2_t0
    transformed2 = alpha21 * k1 * carbon1_t0 - half_difference * carbon2_t0
    eCO2_hat = map(ts.value) do time
        elapsed = time - t0
        gap_time = eigen_gap * elapsed
        common = exp(half_trace * elapsed)
        hyperbolic = sinh(gap_time) / eigen_gap
        carbon1 = common * (
            cosh(gap_time) * carbon1_t0 + hyperbolic * transformed1
        )
        carbon2 = common * (
            cosh(gap_time) * carbon2_t0 + hyperbolic * transformed2
        )
        totalC_t0 - carbon1 - carbon2
    end
    DynamicPPL.@addlogprob! logpdf(Beta(10, 1), gamma)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), k1)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), k2)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), alpha21)
    DynamicPPL.@addlogprob! logpdf(Normal(0, 1), alpha12)
    DynamicPPL.@addlogprob! logpdf(Cauchy(0, 1), sigma)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(Fill(Normal(0, sigma), size(eCO2mean))),
        eCO2mean .- eCO2_hat,
    )
end

function make_model(::Val{Symbol("soil_carbon-soil_incubation")}, data)
    return pdb_soil_incubation(
        data["totalC_t0"], data["t0"], data["N_t"],
        _FixedData(Float64.(data["ts"])), data["eCO2mean"],
    )
end



# PosteriorDB Stan model: state_space_stochastic_level_stochastic_seasonal
@model function pdb_state_space_stochastic_level_stochastic_seasonal(n, y, x, w, seasonal_design, mu_lower, mu_upper)
    mu ~ product_distribution(Fill(Uniform(mu_lower, mu_upper), n))
    seasonal ~ product_distribution(Fill(Flat(), n))
    beta ~ Flat()
    lambda ~ Flat()
    sigma ~ ordered(product_distribution(Fill(FlatPos(0.0), 3)))
            seasonal_location = seasonal_design * seasonal
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Normal(0, sigma[1]), n - 11)),
                seasonal[12:n] .- seasonal_location,
            )
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Normal(0, sigma[2]), n - 1)),
                mu[2:n] .- mu[1:n - 1],
            )
            prediction = mu .+ beta .* x .+ lambda .* w .+ seasonal
            DynamicPPL.@addlogprob! logpdf(
                product_distribution(Fill(Normal(0, sigma[3]), n)),
                y .- prediction,
            )
            DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(LocationScale(0, 1, TDist(4)), size(sigma))), sigma)
end

function make_model(::Val{Symbol("uk_drivers-state_space_stochastic_level_stochastic_seasonal")}, data)
    n = data["n"]
    y = Float64.(data["y"])
    seasonal_design = zeros(n - 11, n)
    for t in 12:n
        seasonal_design[t - 11, t - 11:t - 1] .= -1.0
    end
    mu_lower = mean(y) - 3std(y)
    mu_upper = mean(y) + 3std(y)
    return pdb_state_space_stochastic_level_stochastic_seasonal(
        n, y, Float64.(data["x"]), Float64.(data["w"]), seasonal_design,
        mu_lower, mu_upper,
    )
end



# PosteriorDB Stan model: surgical_model
@model function pdb_surgical_model(N, r, n)
    mu ~ Flat()
    sigmasq ~ FlatPos(0)
    b ~ product_distribution(Fill(Flat(), N))
    sigma = sqrt(sigmasq)
    DynamicPPL.@addlogprob! logpdf(Normal(0.0, 1000.0), mu)
    DynamicPPL.@addlogprob! logpdf(InverseGamma(0.001, 0.001), sigmasq)
    DynamicPPL.@addlogprob! logpdf(product_distribution(Fill(Normal(mu, sigma), size(b))), b)
    DynamicPPL.@addlogprob! logpdf(
        product_distribution(map(BinomialLogit, n, b)), r,
    )
end

function make_model(::Val{Symbol("surgical_data-surgical_model")}, data)
    return pdb_surgical_model(data["N"], data["r"], data["n"])
end



# PosteriorDB Stan model: wells_daae_c_model
@model function pdb_wells_daae_c_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 5))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_daae_c_model")}, data)
    c_dist100 = (data["dist"] .- mean(data["dist"])) ./ 100
    c_arsenic = data["arsenic"] .- mean(data["arsenic"])
    X = hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic,
             data["assoc"], data["educ"] ./ 4)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_daae_c_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_dae_c_model
@model function pdb_wells_dae_c_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 4))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dae_c_model")}, data)
    c_dist100 = (data["dist"] .- mean(data["dist"])) ./ 100
    c_arsenic = data["arsenic"] .- mean(data["arsenic"])
    X = hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, data["educ"] ./ 4)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dae_c_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_dae_inter_model
@model function pdb_wells_dae_inter_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 6))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dae_inter_model")}, data)
    c_dist100 = (data["dist"] .- mean(data["dist"])) ./ 100
    c_arsenic = data["arsenic"] .- mean(data["arsenic"])
    c_educ4 = (data["educ"] .- mean(data["educ"])) ./ 4
    X = hcat(c_dist100, c_arsenic, c_educ4, c_dist100 .* c_arsenic,
             c_dist100 .* c_educ4, c_arsenic .* c_educ4)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dae_inter_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_dae_model
@model function pdb_wells_dae_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 3))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dae_model")}, data)
    X = hcat(data["dist"] ./ 100, data["arsenic"], data["educ"] ./ 4)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dae_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_dist
@model function pdb_wells_dist(signs, X)
    X = X.value
    beta ~ product_distribution(Fill(Flat(), 2))
    logits = beta[1] .+ X * view(beta, 2:2)
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dist")}, data)
    X = reshape(Float64.(data["dist"]), data["N"], 1)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dist(signs, _FixedData(X))
end



# PosteriorDB Stan model: wells_dist100_model
@model function pdb_wells_dist100_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 1))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dist100_model")}, data)
    X = reshape(data["dist"] ./ 100, data["N"], 1)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dist100_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_dist100ars_model
@model function pdb_wells_dist100ars_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 2))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_dist100ars_model")}, data)
    X = hcat(data["dist"] ./ 100, data["arsenic"])
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_dist100ars_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_interaction_c_model
@model function pdb_wells_interaction_c_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 3))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_interaction_c_model")}, data)
    c_dist100 = (data["dist"] .- mean(data["dist"])) ./ 100
    c_arsenic = data["arsenic"] .- mean(data["arsenic"])
    X = hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic)
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_interaction_c_model(signs, _FixedData(Float64.(X)))
end



# PosteriorDB Stan model: wells_interaction_model
@model function pdb_wells_interaction_model(signs, X)
    X = X.value
    alpha ~ Flat()
    beta ~ product_distribution(Fill(Flat(), 3))
    logits = alpha .+ X * beta
    DynamicPPL.@addlogprob! -sum(log1pexp.(signs .* logits))
end

function make_model(::Val{Symbol("wells_data-wells_interaction_model")}, data)
    dist100 = data["dist"] ./ 100
    X = hcat(dist100, data["arsenic"], dist100 .* data["arsenic"])
    signs = 1 .- 2 .* Float64.(data["switched"])
    return pdb_wells_interaction_model(signs, _FixedData(Float64.(X)))
end



# Mooncake rules

import Mooncake

Mooncake.tangent_type(::Type{<:_FixedData}) = Mooncake.NoTangent

Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{
    typeof(logpdf),
    RBFKroneckerNormal{Float64,Matrix{Float64},Matrix{Float64}},
    Matrix{Float64},
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(logpdf)},
    distribution::Mooncake.CoDual{
        <:RBFKroneckerNormal{Float64,Matrix{Float64},Matrix{Float64}}
    },
    y::Mooncake.CoDual{Matrix{Float64}},
)
    d = Mooncake.primal(distribution)
    py = Mooncake.primal(y)

    exponential = exp.(d.distances .* d.bandwidth)
    column_cov = Symmetric(d.variance .* exponential + 1.0e-5I)
    row_cov = Symmetric(d.row_cholesky * d.row_cholesky')
    row_values, row_vectors = eigen(row_cov)
    column_values, column_vectors = eigen(column_cov)
    variances = row_values .* column_values' .+ d.nugget
    rotated = row_vectors' * py * column_vectors
    value = -0.5 * (
        length(py) * log(2pi) +
        sum(abs2.(rotated) ./ variances) +
        sum(log, variances)
    )

    alpha = row_vectors * (rotated ./ variances) * column_vectors'
    column_trace = vec(sum(row_values ./ variances; dims=1))
    row_trace = vec(sum(column_values' ./ variances; dims=2))
    column_gradient = 0.5 .* (
        alpha' * row_cov * alpha -
        column_vectors * Diagonal(column_trace) * column_vectors'
    )
    row_gradient = 0.5 .* (
        alpha * column_cov * alpha' -
        row_vectors * Diagonal(row_trace) * row_vectors'
    )
    variance_gradient = dot(column_gradient, exponential)
    bandwidth_gradient = dot(
        column_gradient,
        d.variance .* exponential .* d.distances,
    )
    cholesky_gradient = 2 .* row_gradient * d.row_cholesky
    nugget_gradient = 0.5 * (sum(abs2, alpha) - sum(inv, variances))

    # `Symmetric(A)` reads A's upper triangle. Fold the covariance derivative
    # into that storage before propagating it through the RBF construction.
    column_storage_gradient = zeros(size(column_gradient))
    for j in axes(column_storage_gradient, 2)
        column_storage_gradient[j, j] = column_gradient[j, j]
        for i in firstindex(column_storage_gradient, 1):(j - 1)
            column_storage_gradient[i, j] =
                column_gradient[i, j] + column_gradient[j, i]
        end
    end
    distances_gradient = column_storage_gradient .* (
        d.variance * d.bandwidth .* exponential
    )

    fields = Mooncake._fields(Mooncake.tangent(distribution))
    function pullback!!(delta)
        fields.row_cholesky .+= delta .* cholesky_gradient
        fields.distances .+= delta .* distances_gradient
        Mooncake.tangent(y) .-= delta .* alpha
        distribution_rdata = Mooncake.RData((
            variance=delta * variance_gradient,
            bandwidth=delta * bandwidth_gradient,
            row_cholesky=Mooncake.NoRData(),
            nugget=delta * nugget_gradient,
            distances=Mooncake.NoRData(),
        ))
        return Mooncake.NoRData(), distribution_rdata, Mooncake.NoRData()
    end
    return Mooncake.zero_fcodual(value), pullback!!
end
