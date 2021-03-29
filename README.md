# DynamicPPL.jl

[![CI](https://github.com/TuringLang/DynamicPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![JuliaNightly](https://github.com/TuringLang/DynamicPPL.jl/workflows/JuliaNightly/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AJuliaNightly+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/DynamicPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/DynamicPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/DynamicPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/DynamicPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/DynamicPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/DynamicPPL.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://colprac.sciml.ai/)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/24589)

A domain-specific language and backend for probabilistic programming languages, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing then please do get in touch and open an issue on Github.

### Contributor's Guide

This project follows the [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://colprac.sciml.ai/), apart from the following slight variation:

- The master branch contains the most recent release at any point in time. All non-breaking changes (bug fixes etc.) are merged directly into master and a new patch version is released immediately.
- A separate dev branch contains all breaking changes, and is merged into master when a minor version release happens.

For instance, suppose we are currently on version 0.13.5.

- If someone produces a bug fix, it is merged directly into master and bumps the version to 0.13.6. This change is also merged into dev so that it remains up-to-date with master.
- If someone is working on a new feature that is not breaking (performance-related, fancy new syntax that is backwards-compatible etc.), the same happens.
- New breaking changes are merged into dev until a release is ready to go, at which point dev is merged into master and version 0.14 is released.

### Bors

This project uses [Bors](https://bors.tech/) for merging PRs. Bors is a Github bot that prevents merge skew / semantic merge conflicts by testing
the exact integration of pull requests before merging them.

When a PR is good enough for merging and has been approved by at least one reviewer, instead of merging immediately, it is added to the merge queue
by commenting with `bors r+`. The Bors bot merges the pull request into a staging area, and runs the CI tests. If tests pass, the commit in the staging
area is copied to the target branch (i.e., usually master).

PRs can be tested by adding a comment with `bors try`. Additional commands can be found in the [Bors documentation](https://bors.tech/documentation/).
