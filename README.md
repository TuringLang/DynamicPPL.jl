# DynamicPPL.jl

[![Build Status](https://travis-ci.com/TuringLang/DynamicPPL.jl.svg?branch=master)](https://travis-ci.com/TuringLang/DynamicPPL.jl)
[![Codecov](https://codecov.io/gh/TuringLang/DynamicPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/DynamicPPL.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

A domain-specific language and backend for probabilistic programming languages, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing then please do get in touch and open an issue on Github.

This project follows the [ColPrac guide on collaborative practices](http://colprac.sciml.ai/), apart from the following slight variation:

- The master branch contains the most recent release at any point in time. All non-breaking changes (bug fixes etc.) are merged directly into master and a new patch version is released immediately.
- A separate dev branch contains all breaking changes, and is merged into master when a minor version release happens.

For instance, suppose we are currently on version 0.13.5.

- If someone produces a bug fix, it is merged directly into master and bumps the version to 0.13.6. This change is also merged into dev so that it remains up-to-date with master.
- If someone is working on a new feature that is not breaking (performance-related, fancy new syntax that is backwards-compatible etc.), the same happens.
- New breaking changes are merged into dev until a release is ready to go, at which point dev is merged into master and version 0.14 is released.
