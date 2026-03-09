# Manipulation of VNTs

Once you have created a VNT (or obtained it from somewhere in DynamicPPL), there are a number of functions which can be used to modify its contents.
These are faithfully documented in the API docs.
Many of them are intended as iteration tools, such as `mapreduce`, and don't really require thorough explanation.

However, here we will briefly discuss two of the more 'special' functions, which modify the representation of a VNT and are used by DynamicPPL in fairly subtle ways.
In particular, they are used extensively in the construction of chain objects from MCMC sampling.

## Densification

The function [`DynamicPPL.densify!!`](@ref) converts VNTs to 'dense' representations: essentially, this means converting any `PartialArray`s whose masks are completely `true` to regular arrays.

For example, this VNT contains a `PartialArray` which is *really* the same thing as a regular array:

```@example 1
using DynamicPPL

vnt = @vnt begin
    @template x = zeros(2)
    x[1] := 1.0
    x[2] := 2.0
end
```

Calling `densify!!` on it leads to:

```@example 1
dense_vnt = densify!!(vnt)
```

Note that `vnt` and `dense_vnt` contain the same values, and you can index into them in the same way:

```@example 1
vnt[@varname(x)], dense_vnt[@varname(x)]
```

However, they differ in the *keys* that they return.

```@example 1
keys(vnt), keys(dense_vnt)
```

In other words, `vnt` retains the notion that `x[1]` and `x[2]` are separate, whereas `dense_vnt` treats `x` as a single entity.
Often the latter is more convenient: consider e.g. sampling from the following model

```@example 1
using Distributions

@model function my_model()
    x = zeros(2)
    return x .~ Normal()
end

# We aren't running this, but in principle, this is what you would do.
# using Turing, FlexiChains
# chain = sample(my_model(), NUTS(), 100; chain_type=VNChain)
```

In the resulting `chain`, if `x[1]` and `x[2]` were stored separately as different matrices, this would be quite annoying for post-processing.
To get a sample of the vector `x`, you would have to manually perform some kind of concatenation, e.g.

```julia
# Again this isn't run.
# cat(chain[@varname(x[1])], chain[@varname(x[2])]; dims=:xindex)
```

Now, notice that when we evaluate the model, we actually get non-dense VNTs:

```@example 1
rand(my_model())
```

However, if we were to actually use Turing and FlexiChains to sample from this model, we will find that FlexiChains stores the samples in a dense format.
This is because inside the constructor of `DynamicPPL.ParamsWithStats`, DynamicPPL calls `densify!!` on the parameter VNTs before handing them off to FlexiChains.

!!! note
    
    In principle, MCMCChains is also given densified VNTs; unfortunately, the first thing that MCMCChains does is to just split them back up into separate scalar-valued parameters, so you don't actually get to see the benefits of densification when using MCMCChains.

Note that densification, and the resulting improvements in chain structure, is something that can only really be done with VNTs as a base data structure.
If we had plain old `OrderedDict`s mapping `x[1] => 1.0` and `x[2] => 2.0`, there would be no way to know that `x` *only* contained these two entries and nothing else.
This goes back to our arguments about 'constructiveness' [in the VNT motivation page](@ref constructiveness).

## Skeleton VNTs

Another function, [`DynamicPPL.skeleton`](@ref), is used to generate 'skeletons' of VNTs.
Specifically, a skeleton of a VNT is one that contains enough template information to reconstruct the VNT from its key-value pairs.

This is best illustrated by example:

```@example 1
vnt = @vnt begin
    # We set x to be length-3 to avoid it ever being
    # densified, which would render this example moot.
    @template x = zeros(3)
    x[1] := 1.0
    x[2] := 2.0
    # Stick in another scalar for illustration.
    y := 3.0
end
```

Suppose we convert this into an `OrderedDict`.
This is a lossy operation, since we lose the information that `x` is a length-3 vector.

```@example 1
od = OrderedDict(pairs(vnt)...)
```

We can't go back from this `OrderedDict` to the original VNT: if we attempt to naively do so, although `y` will be faithfully reconstructed, we will get a `GrowableArray` for `x`.

```@example 1
new_vnt = VarNamedTuple(od)
```

Clearly, what we are lacking is the template information for `x`.
This is where the skeleton comes in:

```@example 1
skel = skeleton(vnt)
```

Notice that in the skeleton, the key `y` has been entirely dropped, since we don't need any template information for it.
Furthermore, the PartialArray `x` has been replaced with a *normal* array with the same size and array type (i.e. `Base.Array`), but with all values set to `nothing`.

Here, `nothing` is chosen because storing it in an array is extremely space-efficient:

```@example 1
Base.sizeof(fill(nothing, 100))
```

Furthermore, the element type of the array is not needed when reconstructing data.
For example, this is how we would reconstruct the original VNT:

```@example 1
using AbstractPPL: getsym

# In a real setting you wouldn't need `begin .. end` and `local`.
begin
    local new_vnt = VarNamedTuple()
    for (vn, val) in pairs(od)
        top_sym = getsym(vn)
        template = get(skel.data, top_sym, DynamicPPL.NoTemplate())
        new_vnt = DynamicPPL.templated_setindex!!(new_vnt, val, vn, template)
    end
    new_vnt
end
```

In the call to `templated_setindex!!` for `x`, the template is `fill(nothing, 3)`.
However, the element type of this template array is not important: when DynamicPPL instantiates the new PartialArray, it will use `typeof(val)` rather than `eltype(template)` to determine the element type it should use.
This allows us to freely choose any element type we like in the skeleton.

Skeleton VNTs are not currently used in DynamicPPL.
However, it is likely that they will find some kind of use in the future as a minimalistic representation of the *structure* of a VNT, without any of the actual values.
For example, consider model conditioning:

```@example 1
@model function f()
    x = zeros(2)
    return x .~ Normal()
end

cond_model = f() | Dict(@varname(x[1]) => 1.0, @varname(x[2]) => 2.0)
```

Right now, conditioning with a `Dict` will lead to the conditioning values being stored with `GrowableArray`s, which is not ideal.
However, if the *model* carried with itself a skeleton VNT, then at the point where we condition the model, we could use that to reconstruct a VNT of conditioned values.
This is not yet implemented, but is one of the use cases that we have in mind.

FlexiChains does currently make use of skeleton VNTs.
If we reevaluate a model with a `VNChain`, we need some way of reconstructing VNTs to feed into the model.
However, FlexiChains stores parameters in (more or less) an `OrderedDict{VarName,Matrix}`, from which one cannot reconstruct a VNT (you can only reconstruct an `OrderedDict{VarName}`).
To get around this, FlexiChains additionally stores the skeleton VNT which bridges this gap and allows e.g. `FlexiChains.parameters_at` to return a VNT.
Please see the [FlexiChains documentation](https://pysm.dev/FlexiChains.jl) for more details on this.
