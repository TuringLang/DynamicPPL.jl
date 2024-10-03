# DynamicPPL.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/DynamicPPL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/DynamicPPL.jl/dev)
[![CI](https://github.com/TuringLang/DynamicPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![JuliaPre](https://github.com/TuringLang/DynamicPPL.jl/workflows/JuliaPre/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AJuliaPre+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/DynamicPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/DynamicPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/DynamicPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/DynamicPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/DynamicPPL.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://docs.sciml.ai/ColPrac/stable/)

*A domain-specific language and backend for probabilistic programming, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).*

DynamicPPL is the part of Turing.jl that deals with defining, running, and manipulating models. DynamicPPL provides:

  - General-purpose probabilistic programming with an intuitive syntax.
  - The `@model` syntax and macro for easily specifying probabilistic generative models.
  - A tracing data-structure for tracking random variables in dynamic probabilistic models.
  - A rich contextual dispatch system allowing for tailored behaviour during model execution.
  - A user-friendly syntax for probabilistic queries.

Information on how to use the DynamicPPL frontend to build Bayesian models can be found on the [Turing website](https://turinglang.org/). Tutorials explaining how to use the backend can be found [alongside the documentation](https://turinglang.github.io/DynamicPPL.jl/stable/). More information can be found in our paper [DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models](https://arxiv.org/pdf/2002.02702.pdf).

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing, please get in touch! You can find us in the #turing channel on the [Julia Slack](https://julialang.org/slack/) or [Discourse](discourse.julialang.org). If you're having any problems, please open a Github issue, even if the problem seems small (like help figuring out an error message). Every issue you open helps us improve the library!

### Contributor's Guide

This project follows the [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://docs.sciml.ai/ColPrac/stable/).

### Merge Queue

This project uses a [merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue) for merging PRs.
In this way, merge skew / semantic merge conflicts are prevented by testing the exact integration of pull requests before merging them.

When a PR is good enough for merging and has been approved by at least one reviewer, instead of merging immediately, it is added to the merge queue.
If the CI tests pass, including downstream tests of Turing, the PR is merged into the main branch.
