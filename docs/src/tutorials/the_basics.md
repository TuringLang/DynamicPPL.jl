```@raw html
<style>
    table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    pre, div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "95b4dca10a6841e1b3f8b3df89f1964e046906e0cc1e126443b841140d84419a"
    julia_version = "1.7.2"
-->

<div class="markdown"><h1>The Basics</h1>
</div>


<div class="markdown"><p>DynamicPPL is a probabilistic programming language embedded in Julia. You can think of it as the backend that powers Turing, doing things like:</p>
<ul>
<li><p>Converting your readable Turing code into runnable Julia code with the <code>@model</code> macro.</p>
</li>
<li><p>Providing information to samplers about a model.</p>
</li>
<li><p>Keeping track of random variables while sampling.</p>
</li>
<li><p>Calculating the log-pdf for possible events.</p>
</li>
</ul>
<p>In this tutorial, we&#39;ll see how to do all of this. Let&#39;s start by creating a basic model.</p>
</div>

<pre class='language-julia'><code class='language-julia'>using DynamicPPL, Distributions, Random</code></pre>


<pre class='language-julia'><code class='language-julia'>rng = Xoshiro(1776);  # Set seed for reproducibility</code></pre>


<pre class='language-julia'><code class='language-julia'># This model assumes that our sample follows a normal distribution with a standard deviation of 1 and an unknown mean.
@model function demo(x)
   μ ~ Normal(0, 1)  # Our prior is a standard normal distribution
   x .~ Normal(μ, 1)
   return nothing
end</code></pre>
<pre id='var-demo' class='pre-class'><code class='code-output'>demo (generic function with 2 methods)</code></pre>

<pre class='language-julia'><code class='language-julia'># We instantiate the model by calling it on a dataset.
dataset = [-1, 1, 1]; model = demo(dataset)</code></pre>
<pre id='var-dataset' class='pre-class'><code class='code-output'>Model{typeof(demo), (:x,), (), (), Tuple{Vector{Int64}}, Tuple{}, DefaultContext}(:demo, Main.workspace#16.demo, (x = [-1, 1, 1],), NamedTuple(), DefaultContext())</code></pre>


<div class="markdown"><p>In our case, we can calculate the posterior analytically. Let&#39;s do that so we can compare results later:</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin posterior = Normal(
	mean(vcat(dataset, 0)), 
	1 / sqrt(length(vcat(dataset, 0)))
)
end</code></pre>
<pre id='var-posterior' class='pre-class'><code class='code-output'>Normal{Float64}(μ=0.25, σ=0.5)</code></pre>


<div class="markdown"><h3>Evaluating a model</h3>
<p>What <em>is</em> a DynamicPPL model? This is an important question, because the way DynamicPPL treats models can be different from how you&#39;d think of them.</p>
<p>DynamicPPL is <em>procedural</em>: everything is a series of instructions &#40;a procedure&#41;. This procedure modifies or returns <code>AbstractVarInfo</code> objects, which hold samples taken from a probability distribution. As an example, a <code>SimpleVarInfo</code> has two fields, a <code>NamedTuple</code> of variables and a log-probability:</p>
</div>

<pre class='language-julia'><code class='language-julia'>x0 = SimpleVarInfo((μ=0,), logpdf(posterior, 0))</code></pre>
<pre id='var-x0' class='pre-class'><code class='code-output'>SimpleVarInfo((μ = 0,), -0.3507913526447274)</code></pre>


<div class="markdown"><p>The procedural approach is different from the object-oriented approach. We don&#39;t try to reason through a model by thinking of each <code>~</code> as a property of a model object, or think of the model as a coherent &quot;Whole&quot; to be queried. A <code>~</code> is an <em>action</em>.</p>
<p>We can see this by taking a look at what&#39;s inside our model. A <code>Model</code> is pretty much just a wrapper around a function <code>f</code>, with hardly any methods or fields:</p>
</div>

<pre class='language-julia'><code class='language-julia'>fieldnames(DynamicPPL.Model)</code></pre>
<pre id='var-hash177926' class='pre-class'><code class='code-output'>(:name, :f, :args, :defaults, :context)</code></pre>


<div class="markdown"><p>Procedural programming is also very different from functional programming, where we might think of a model as a pure mathematical function, like a probability density. Each <code>~</code> statement in DynamicPPL is not a pure function: lines can behave differently depending on the model&#39;s state. This state is contained in the <code>Context</code> and <code>Sampler</code> arguments we pass to a model.</p>
<p>We can execute a model using <code>DynamicPPL.evaluate&#33;&#33;</code>. This will always return two values: the return value for the function &#40;in our case <code>nothing</code>&#41; and a <code>VarInfo</code>. &#40;It will also modify <code>VarInfo</code> if <code>VarInfo</code> is mutable.&#41;</p>
<p>Let&#39;s run through the most important DynamicPPL <code>Context</code>s. First we have the <code>SamplingContext</code>, which draws a new <code>VarInfo</code> using the given sampler:</p>
</div>

<pre class='language-julia'><code class='language-julia'># Here, our sampler is SampleFromPrior()
_, x1 = DynamicPPL.evaluate!!(model, x0, SamplingContext())</code></pre>
<pre id='var-x1' class='pre-class'><code class='code-output'>(nothing, SimpleVarInfo{NamedTuple{(:μ,), Tuple{Float64}}, Float64}((μ = -0.8169973995193072,), -7.32772103398062))</code></pre>


<div class="markdown"><p>On the other hand, if we call a model with a <code>LikelihoodContext</code> and a preexisting <code>VarInfo</code>, the model evaluates the likelihood function &#40;ignoring the prior&#41; and inserts it into the <code>logp</code> field, leaving the sample unchanged:</p>
</div>

<pre class='language-julia'><code class='language-julia'>_, x2 = DynamicPPL.evaluate!!(model, deepcopy(x1), LikelihoodContext())</code></pre>
<pre id='var-x2' class='pre-class'><code class='code-output'>(nothing, SimpleVarInfo{NamedTuple{(:μ,), Tuple{Float64}}, Float64}((μ = -0.8169973995193072,), -6.075040125365291))</code></pre>


<div class="markdown"><p>And the value of <code>logp</code> for <code>x1</code> is equal to the likelihood plus the prior:</p>
</div>

<pre class='language-julia'><code class='language-julia'>_, x3 = DynamicPPL.evaluate!!(model, deepcopy(x1), PriorContext())</code></pre>
<pre id='var-x3' class='pre-class'><code class='code-output'>(nothing, SimpleVarInfo{NamedTuple{(:μ,), Tuple{Float64}}, Float64}((μ = -0.8169973995193072,), -1.252680908615328))</code></pre>

<pre class='language-julia'><code class='language-julia'>getlogp(x1) ≈ getlogp(x2) + getlogp(x3)</code></pre>
<pre id='var-hash170254' class='pre-class'><code class='code-output'>true</code></pre>


<div class="markdown"><p>For convenience, we also provide the functions <code>logprior</code>, <code>loglikelihood</code>, and <code>logjoint</code> to calculate probabilities for a <code>VarInfo</code>:</p>
</div>

<pre class='language-julia'><code class='language-julia'>logjoint(model, x1) ≈ loglikelihood(model, x1) + logprior(model, x1)</code></pre>
<pre id='var-hash206004' class='pre-class'><code class='code-output'>true</code></pre>


<div class="markdown"><p>Some contexts can be nested. For instance, <code>SamplingContext</code> can be nested with a <code>PriorContext</code> to insert the log-prior, rather than the log-posterior into <code>logp</code>.</p>
</div>

<pre class='language-julia'><code class='language-julia'>_, x4 = DynamicPPL.evaluate!!(model, x1, SamplingContext())</code></pre>
<pre id='var-x4' class='pre-class'><code class='code-output'>(nothing, SimpleVarInfo{NamedTuple{(:μ,), Tuple{Float64}}, Float64}((μ = -2.5499741548354127,), -20.73046466831126))</code></pre>


<div class="markdown"><p>By default, we evaluate the log-posterior. This can be specified explicitly using <code>DefaultContext&#40;&#41;</code>.</p>
</div>

<pre class='language-julia'><code class='language-julia'>_, x5 = DynamicPPL.evaluate!!(model, deepcopy(x1), DefaultContext()); x5 == x1</code></pre>
<pre id='var-x5' class='pre-class'><code class='code-output'>true</code></pre>


<div class="markdown"><h3>Example: A simple sampler &#40;say 5 times fast&#41;</h3>
</div>


<div class="markdown"><p>Let&#39;s create a Metropolis-Hastings Sampler to see how this works:</p>
</div>

<pre class='language-julia'><code class='language-julia'>function sample(rng, model::DynamicPPL.Model, kernel::Distribution, n_steps::Int)
	# First we create a SimpleVarInfo by sampling from the prior, to initialize the model. For convenience, SimpleVarInfo(model) samples from the prior by default.
	init = SimpleVarInfo(model)
	# Now we use a function barrier to let Julia infer the correct types for `vi` -- if we don't include one, `vi` may be slow.
	return metropolis(rng, init, model, kernel, n_steps)
end</code></pre>
<pre id='var-sample' class='pre-class'><code class='code-output'>sample (generic function with 1 method)</code></pre>

<pre class='language-julia'><code class='language-julia'>function metropolis(
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
		accept = randexp(rng) &gt; log_p
		
		samples[step] = accept ? proposal : current
		current = samples[step]
	end
	return samples
end</code></pre>
<pre id='var-metropolis' class='pre-class'><code class='code-output'>metropolis (generic function with 1 method)</code></pre>

<pre class='language-julia'><code class='language-julia'>samples = sample(rng, model, Normal(0, .05), 1_000_000)</code></pre>
<pre id='var-samples' class='pre-class'><code class='code-output'>1000000-element Vector{SimpleVarInfo{NamedTuple{(:μ,), Tuple{Float64}}, Float64}}:
 SimpleVarInfo((μ = 0.29562447208698855,), -5.054917317725124)
 SimpleVarInfo((μ = 0.24054918806509884,), -5.0509327685111485)
 SimpleVarInfo((μ = 0.25484192626895985,), -5.050801021318679)
 SimpleVarInfo((μ = 0.360609163103266,), -5.075222906743502)
 SimpleVarInfo((μ = 0.4356473248988908,), -5.119683991302919)
 SimpleVarInfo((μ = 0.43813405233515335,), -5.1215429761147835)
 SimpleVarInfo((μ = 0.46242827247081114,), -5.141005674708557)
 ⋮
 SimpleVarInfo((μ = 0.1591200509263067,), -5.067272463105966)
 SimpleVarInfo((μ = 0.07068034752347102,), -5.115065208347296)
 SimpleVarInfo((μ = 0.08359900724689118,), -5.106132713597131)
 SimpleVarInfo((μ = 0.05299550090413724,), -5.128375678146715)
 SimpleVarInfo((μ = 0.03677312272125808,), -5.141685535206778)
 SimpleVarInfo((μ = 0.05665848808243569,), -5.125516013279831)</code></pre>


<div class="markdown"><p>And now we can see we have the right mean&#33;</p>
</div>

<pre class='language-julia'><code class='language-julia'>means = getindex.(samples, (@varname(μ),))</code></pre>
<pre id='var-means' class='pre-class'><code class='code-output'>1000000-element Vector{Float64}:
 0.29562447208698855
 0.24054918806509884
 0.25484192626895985
 0.360609163103266
 0.4356473248988908
 0.43813405233515335
 0.46242827247081114
 ⋮
 0.1591200509263067
 0.07068034752347102
 0.08359900724689118
 0.05299550090413724
 0.03677312272125808
 0.05665848808243569</code></pre>

<pre class='language-julia'><code class='language-julia'>mean(means)</code></pre>
<pre id='var-hash285580' class='pre-class'><code class='code-output'>0.24922598038351465</code></pre>

<!-- PlutoStaticHTML.End -->
```