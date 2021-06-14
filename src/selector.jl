struct Selector
    gid::UInt64
    tag::Symbol # :default, :invalid, :Gibbs, :HMC, etc.
    rerun::Bool
end
function Selector(tag::Symbol=:default, rerun=tag != :default)
    return Selector(time_ns(), tag, rerun)
end
function Selector(gid::Integer, tag::Symbol=:default)
    return Selector(gid, tag, tag != :default)
end
hash(s::Selector) = hash(s.gid)
==(s1::Selector, s2::Selector) = s1.gid == s2.gid
