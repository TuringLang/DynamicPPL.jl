# Pretty-printing for VarNamedTuple and PartialArray.

const ELIDED_COLOR = :light_black

function Base.show(io::IO, vnt::VarNamedTuple)
    if isempty(vnt.data)
        return print(io, "VarNamedTuple()")
    end
    print(io, "VarNamedTuple")
    show(io, vnt.data)
    return nothing
end

const MAX_KEYS_OR_INDICES = 8

function vnt_pretty_print(io::IO, pa::PartialArray, prefix::String, depth::Int)
    # an incomplete PA is dimmed to indicate that it has missing entries.
    size_style = if all(pa.mask)
        (;)
    else
        (; color=ELIDED_COLOR)
    end
    print(io, "PartialArray ")
    printstyled(io, "size=" * string(size(pa.data)); size_style...)
    println(io, " data::" * string(typeof(pa.data)))
    # can't use `keys(pa)` here as that recurses into each element.
    nindices = count(pa.mask)
    truncate = nindices > MAX_KEYS_OR_INDICES
    active_indices = CartesianIndices(pa.data)[pa.mask]
    for (i, idx) in enumerate(active_indices)
        tree_symbol = i == nindices ? "└─ " : "├─ "
        if truncate && i > 2 && i < nindices
            if i == 3
                print(io, prefix * "│  ")
                printstyled(
                    io,
                    "⋮ (" * string(nindices - 3) * " more set indices) ";
                    color=ELIDED_COLOR,
                )
                println(io)
            end
            continue
        end
        key_name = string(Tuple(idx))
        print(io, prefix * tree_symbol)
        printstyled(io, key_name; color=color_at_depth(depth))
        print(io, " => ")
        v = pa.data[idx]
        if v isa VarNamedTuple || v isa PartialArray
            nspaces_for_key = length(key_name) + 4
            vnt_pretty_print(
                io,
                v,
                prefix * (i == nindices ? "   " : "│  ") * (" "^nspaces_for_key),
                depth + 1,
            )
        else
            show(io, v)
        end
        if i < nindices
            println(io)
        end
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/plain", pa::PartialArray)
    return vnt_pretty_print(io, pa, "", 0)
end

colors = [:red, :green, :blue, :yellow, :magenta, :cyan]
color_at_depth(depth::Int) = colors[mod1(depth, length(colors))]

function vnt_pretty_print(io::IO, vnt::VarNamedTuple, prefix::String, depth::Int)
    println(io, "VarNamedTuple")
    nkeys = length(keys(vnt.data))
    # If there are loads of keys, show only the first two and the last
    truncate = nkeys > MAX_KEYS_OR_INDICES
    for (i, (k, v)) in enumerate(zip(keys(vnt.data), values(vnt.data)))
        tree_symbol = i == nkeys ? "└─ " : "├─ "
        if truncate && i > 2 && i < nkeys
            if i == 3
                print(io, prefix * "│  ")
                printstyled(
                    io, "⋮ (" * string(nkeys - 3) * " more keys) "; color=ELIDED_COLOR
                )
                println(io)
            end
            continue
        end
        key_name = string(k)
        print(io, prefix * tree_symbol)
        printstyled(io, key_name; color=color_at_depth(depth))
        print(io, " => ")
        if v isa VarNamedTuple || v isa PartialArray
            nspaces_for_key = length(key_name) + 4
            vnt_pretty_print(
                io,
                v,
                prefix * (i == nkeys ? "   " : "│  ") * (" "^nspaces_for_key),
                depth + 1,
            )
        else
            show(io, v)
        end
        if i < nkeys
            println(io)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", vnt::VarNamedTuple)
    if isempty(vnt.data)
        print(io, "VarNamedTuple()")
    else
        vnt_pretty_print(io, vnt, "", 0)
    end
    return nothing
end
