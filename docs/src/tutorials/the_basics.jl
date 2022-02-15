### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ bfe62f45-7189-476b-8b43-81e806246c7d
using DynamicPPL, Distributions, Random

# ╔═╡ c97e131f-3157-42d9-8ba5-5b07b96dd661
md"""
# The Basics
"""

# ╔═╡ 256d7c45-4b33-4778-8b4e-36e0544fef2f
md"""
DynamicPPL is a probabilistic programming language embedded in Julia. You can think of it as the backend that powers Turing, doing things like:
* Converting your readable Turing code into runnable Julia code with the `@model` macro.
* Providing information to samplers about a model.
* Keeping track of random variables while sampling.
* Calculating the log-pdf for possible events.

In this tutorial, we'll see how to do all of this. Let's start by creating a basic model.
"""

# ╔═╡ 122fac69-13a1-4b27-b7c6-0c95c4a47255
rng = Xoshiro(1776);  # Set seed for reproducibility

# ╔═╡ b6e9e832-04c1-455f-a560-bb0c20e49648
# This model assumes that our sample follows a normal distribution with a standard deviation of 1 and an unknown mean.
@model function demo(x)
   μ ~ Normal(0, 1)  # Our prior is a standard normal distribution
   x .~ Normal(μ, 1)
   return nothing
end

# ╔═╡ 42dcc0b6-c91f-4d25-a1b7-c648541fdd82
# We instantiate the model by calling it on a dataset.
dataset = [-1, 1, 1]; model = demo(dataset)

# ╔═╡ 0608af06-d5fe-4382-b3b0-db666ec8301d
md"""
In our case, we can calculate the posterior analytically. Let's do that so we can compare results later:
"""

# ╔═╡ 2c65dc36-d4aa-4099-a2c1-975450bb8f85
begin posterior = Normal(
	mean(vcat(dataset, 0)), 
	1 / sqrt(length(vcat(dataset, 0)))
)
end

# ╔═╡ e8a933a5-524b-46d1-9413-c2de0f61054a
md"""
### Evaluating a model

What *is* a DynamicPPL model? This is an important question, because the way DynamicPPL treats models can be different from how you'd think of them.

DynamicPPL is *procedural*: everything is a series of instructions (a procedure). This procedure modifies or returns `AbstractVarInfo` objects, which hold samples taken from a probability distribution. As an example, a `SimpleVarInfo` has two fields, a `NamedTuple` of variables and a log-probability:
"""

# ╔═╡ 500b1ac1-1753-410f-8c70-6c9034715113
x0 = SimpleVarInfo((μ=0,), logpdf(posterior, 0))

# ╔═╡ d644589a-1b08-48b3-9929-5a14fc1a90a2
md"""
The procedural approach is different from the object-oriented approach. We don't try to reason through a model by thinking of each `~` as a property of a model object, or think of the model as a coherent "Whole" to be queried. A `~` is an *action*.

We can see this by taking a look at what's inside our model. A `Model` is pretty much just a wrapper around a function `f`, with hardly any methods or fields:
"""

# ╔═╡ 9d8677d4-baad-46c7-b440-9d25b55448dd
fieldnames(DynamicPPL.Model)

# ╔═╡ bfd436ef-8f6b-4461-8110-6d58aee010a0
md"""
Procedural programming is also very different from functional programming, where we might think of a model as a pure mathematical function, like a probability density. Each `~` statement in DynamicPPL is not a pure function: lines can behave differently depending on the model's state. This state is contained in the `Context` and `Sampler` arguments we pass to a model.

We can execute a model using `DynamicPPL.evaluate!!`. This will always return two values: the return value for the function (in our case `nothing`) and a `VarInfo`. (It will also modify `VarInfo` if `VarInfo` is mutable.)

Let's run through the most important DynamicPPL `Context`s. First we have the `SamplingContext`, which draws a new `VarInfo` using the given sampler:
"""

# ╔═╡ e7b41916-7749-4c60-9075-52be202278aa
# Here, our sampler is SampleFromPrior()
_, x1 = DynamicPPL.evaluate!!(model, x0, SamplingContext())

# ╔═╡ eba2226c-e517-46a7-83c3-5129e175f07d
md"""
On the other hand, if we call a model with a `LikelihoodContext` and a preexisting `VarInfo`, the model evaluates the likelihood function (ignoring the prior) and inserts it into the `logp` field, leaving the sample unchanged:
"""

# ╔═╡ 1b73c6fe-32ae-49c4-b20f-0b70f6bf43ea
_, x2 = DynamicPPL.evaluate!!(model, deepcopy(x1), LikelihoodContext())

# ╔═╡ cca42fe2-fe87-4c7a-8f59-b38a0d888a0d
md"""
And the value of `logp` for `x1` is equal to the likelihood plus the prior:
"""

# ╔═╡ 08f2e20f-d25c-4407-9c60-54d994ce8611
_, x3 = DynamicPPL.evaluate!!(model, deepcopy(x1), PriorContext())

# ╔═╡ 852a11c8-528d-4792-98c1-b94ced6149bb
getlogp(x1) ≈ getlogp(x2) + getlogp(x3)

# ╔═╡ e8df24e1-3701-4743-8b6e-c8a07e46a0df
md"""
For convenience, we also provide the functions `logprior`, `loglikelihood`, and `logjoint` to calculate probabilities for a `VarInfo`:
"""

# ╔═╡ 558d6bc2-10bc-4729-b5fc-3778d7a8e1a0
logjoint(model, x1) ≈ loglikelihood(model, x1) + logprior(model, x1)

# ╔═╡ 08a5b22b-639e-49c0-bda3-88a46a2c67f9
md"""
Some contexts can be nested. For instance, `SamplingContext` can be nested with a `PriorContext` to insert the log-prior, rather than the log-posterior into `logp`.
"""

# ╔═╡ 2cd6d576-2de0-46e5-8a14-d89e73e3cefe
_, x4 = DynamicPPL.evaluate!!(model, x1, SamplingContext())

# ╔═╡ 5dc4cfe1-6f81-48a8-9661-c57bfe2617ec
md"""
By default, we evaluate the log-posterior. This can be specified explicitly using `DefaultContext()`.
"""

# ╔═╡ 6cea2b7e-1142-44f2-ae51-6850f83bbf12
_, x5 = DynamicPPL.evaluate!!(model, deepcopy(x1), DefaultContext()); x5 == x1

# ╔═╡ 97f5bff2-1111-4b94-bb83-195cddb626aa
md"""
### Example: A simple sampler (say 5 times fast)"""

# ╔═╡ c3263293-f928-429d-85fb-b07ef86f5825
md"""
Let's create a Metropolis-Hastings Sampler to see how this works:
"""

# ╔═╡ c19624ce-bd5e-4ee3-8a9d-796f3521ef24
function metropolis(
	rng,
	init::DynamicPPL.AbstractVarInfo, 
	model::DynamicPPL.Model, 
	kernel::Distribution, 
	n_steps::Int
)
	samples = Vector{typeof(init)}(undef, n_steps)
	current = init
	for step in 1:n_steps
		proposal = map(current.values) do x
			x + rand(rng, kernel)
		end
		proposal = SimpleVarInfo(proposal)
		_, proposal = DynamicPPL.evaluate!!(model, proposal, DefaultContext())
		
		log_p = getlogp(current) - getlogp(proposal)
		accept = randexp(rng) > log_p
		
		samples[step] = accept ? proposal : current
		current = samples[step]
	end
	return samples
end

# ╔═╡ 18351328-7fc1-4d02-8949-9dd7b000baa1
function sample(rng, model::DynamicPPL.Model, kernel::Distribution, n_steps::Int)
	# First we create a SimpleVarInfo by sampling from the prior, to initialize the model. For convenience, SimpleVarInfo(model) samples from the prior by default.
	init = SimpleVarInfo(model)
	# Now we use a function barrier to let Julia infer the correct types for `vi` -- if we don't include one, `vi` may be slow.
	return metropolis(rng, init, model, kernel, n_steps)
end

# ╔═╡ c6320d1a-b75f-452b-95de-ac8c323c4629
samples = sample(rng, model, Normal(0, .05), 1_000_000)

# ╔═╡ fd33741e-e8db-4a5e-a143-c97861af5ed0
md"""
And now we can see we have the right mean!
"""

# ╔═╡ e5f7c7b6-c257-4886-a339-f4a4d5dbd8f6
means = getindex.(samples, (@varname(μ),))

# ╔═╡ 54b5eb55-2b78-41cc-b0d1-026b5a47d6f7
mean(means)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.41"
DynamicPPL = "~0.17.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "db0a7ff3fbd987055c43b4e12d2fa30aaae8749c"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.1"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "Setfield"]
git-tree-sha1 = "fc7080c5807afd5bb95f320e1202f218a0bcc562"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.3.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "a33794b483965bf49deaeec110378640609062b1"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.34"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "36757cd97fb240dc5d1cd8aa1b286e955c392bf1"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "54fc4400de6e5c3e27be6047da2ef6ba355511f8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5863b0b10512ed4add2b5ec07e335dc6121065a5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.41"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "LinearAlgebra", "MacroTools", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "807d28d2cf888ca1bd8aa4786fd19fcbb2f6ad7d"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.17.3"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "0abe7fc220977da88ad86d339335a4517944fea2"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.14"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e6bf188613555c78062842777b116905a9f9dd49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "3f0945b47207a41946baee6d1385e4ca738c25f7"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.68"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─c97e131f-3157-42d9-8ba5-5b07b96dd661
# ╟─256d7c45-4b33-4778-8b4e-36e0544fef2f
# ╠═bfe62f45-7189-476b-8b43-81e806246c7d
# ╠═122fac69-13a1-4b27-b7c6-0c95c4a47255
# ╠═b6e9e832-04c1-455f-a560-bb0c20e49648
# ╠═42dcc0b6-c91f-4d25-a1b7-c648541fdd82
# ╟─0608af06-d5fe-4382-b3b0-db666ec8301d
# ╠═2c65dc36-d4aa-4099-a2c1-975450bb8f85
# ╟─e8a933a5-524b-46d1-9413-c2de0f61054a
# ╠═500b1ac1-1753-410f-8c70-6c9034715113
# ╟─d644589a-1b08-48b3-9929-5a14fc1a90a2
# ╠═9d8677d4-baad-46c7-b440-9d25b55448dd
# ╟─bfd436ef-8f6b-4461-8110-6d58aee010a0
# ╠═e7b41916-7749-4c60-9075-52be202278aa
# ╟─eba2226c-e517-46a7-83c3-5129e175f07d
# ╠═1b73c6fe-32ae-49c4-b20f-0b70f6bf43ea
# ╟─cca42fe2-fe87-4c7a-8f59-b38a0d888a0d
# ╠═08f2e20f-d25c-4407-9c60-54d994ce8611
# ╠═852a11c8-528d-4792-98c1-b94ced6149bb
# ╟─e8df24e1-3701-4743-8b6e-c8a07e46a0df
# ╠═558d6bc2-10bc-4729-b5fc-3778d7a8e1a0
# ╟─08a5b22b-639e-49c0-bda3-88a46a2c67f9
# ╠═2cd6d576-2de0-46e5-8a14-d89e73e3cefe
# ╟─5dc4cfe1-6f81-48a8-9661-c57bfe2617ec
# ╠═6cea2b7e-1142-44f2-ae51-6850f83bbf12
# ╟─97f5bff2-1111-4b94-bb83-195cddb626aa
# ╟─c3263293-f928-429d-85fb-b07ef86f5825
# ╠═18351328-7fc1-4d02-8949-9dd7b000baa1
# ╠═c19624ce-bd5e-4ee3-8a9d-796f3521ef24
# ╠═c6320d1a-b75f-452b-95de-ac8c323c4629
# ╟─fd33741e-e8db-4a5e-a143-c97861af5ed0
# ╠═e5f7c7b6-c257-4886-a339-f4a4d5dbd8f6
# ╠═54b5eb55-2b78-41cc-b0d1-026b5a47d6f7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
