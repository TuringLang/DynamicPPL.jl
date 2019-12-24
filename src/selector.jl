import Base: ==

struct Selector
    gid :: UInt64
    tag :: Symbol # :default, :invalid, :Gibbs, :HMC, etc.
end
Selector() = Selector(time_ns(), :default)
Selector(tag::Symbol) = Selector(time_ns(), tag)
hash(s::Selector) = hash(s.gid)
==(s1::Selector, s2::Selector) = s1.gid == s2.gid
