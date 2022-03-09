# DynamicPPL

*A domain-specific language and backend for probabilistic programming, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).*

DynamicPPL is the part of Turing.jl that deals with defining, running, and manipulating models. DynamicPPL provides:
* General-purpose probabilistic programming with an intuitive modelling interface.
* The `@model` syntax and macro for easily specifying probabilistic generative models.
* A tracing data-structure for tracking random variables in dynamic probabilistic models.
* A rich contextual dispatch system allowing for tailored behaviour during model execution.
* A user-friendly syntax for probabilistic queries.

Information on how to use the DynamicPPL frontend to build Bayesian models can be found on the [Turing website](https://turing.ml/). This documentation explains how to use the DynamicPPL backend to query models.

More information can be found in our paper [DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models](https://arxiv.org/pdf/2002.02702.pdf).
