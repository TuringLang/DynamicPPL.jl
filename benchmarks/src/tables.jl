using BenchmarkTools, Tables, PrettyTables, DrWatson

#######################
# `TrialJudgementRow` #
#######################
struct TrialJudgementRow{names,Textras} <: Tables.AbstractRow
    group::String
    judgement::BenchmarkTools.TrialJudgement
    extras::NamedTuple{names,Textras}
end

function TrialJudgementRow(group::String, judgement::BenchmarkTools.TrialJudgement)
    return TrialJudgementRow(group, judgement, NamedTuple())
end

function Tables.columnnames(::Type{TrialJudgementRow})
    return (
        :group,
        :time,
        :time_judgement,
        :gctime,
        :memory,
        :memory_judgement,
        :allocs,
        :time_tolerance,
        :memory_tolerance,
    )
end
# Dispatch needs to include all type-parameters because Tables.jl is a bit too aggressive
# when it comes to overloading this.
function Tables.columnnames(::Type{TrialJudgementRow{names,Textras}}) where {names,Textras}
    return (Tables.columnnames(TrialJudgementRow)..., names...)
end
Tables.columnnames(row::TrialJudgementRow) = Tables.columnnames(typeof(row))
function Tables.getcolumn(row::TrialJudgementRow, i::Int)
    return Tables.getcolumn(row, Tables.columnnames(row)[i])
end
function Tables.getcolumn(row::TrialJudgementRow, name::Symbol)
    # NOTE: We need to use `getfield` below because `getproperty` is overloaded by Tables.jl
    # and so we'll get a `StackOverflowError` if we try to do something like `row.group`.
    return if name === :group
        getfield(row, name)
    elseif name in Tables.columnnames(TrialJudgementRow)
        # `name` is one of the default columns
        j = getfield(row, :judgement)
        if name === :time_judgement
            j.time
        elseif name === :memory_judgement
            j.memory
        elseif name === :time_tolerance
            params(j).time_tolerance
        elseif name === :memory_tolerance
            params(j).memory_tolerance
        else
            # Defer the rest to the `TrialRatio`.
            r = j.ratio
            getfield(r, name)
        end
    else
        # One of `row.extras`
        extras = getfield(row, :extras)
        getfield(extras, name)
    end
end

Tables.istable(rows::Vector{<:TrialJudgementRow}) = true

Tables.rows(rows::Vector{<:TrialJudgementRow}) = rows
Tables.rowaccess(rows::Vector{<:TrialJudgementRow}) = true

# Because DataFrames.jl doesn't respect the `columnaccess`:
# https://github.com/JuliaData/DataFrames.jl/blob/2b9f6673547259bab9fb3bf3b5224eebc7b11ecd/src/other/tables.jl#L48-L61.
Tables.columnaccess(rows::Vector{<:TrialJudgementRow}) = true
function Tables.columns(rows::Vector{<:TrialJudgementRow})
    return (;
        ((name, getproperty.(rows, name)) for name in Tables.columnnames(eltype(rows)))...
    )
end

#########################
# PrettyTables.jl usage #
#########################
function make_highlighter_judgement(isgood)
    function highlighter_judgement(data::Vector{<:TrialJudgementRow}, i, j)
        names = Tables.columnnames(eltype(data))
        name = names[j]
        row = data[i]
        x = row[j]

        if name === :time || name === :time_judgement
            j = row[:time_judgement]
            if j === :improvement
                return isgood
            elseif j === :regression
                return !isgood
            end
        elseif name === :memory || name === :memory_judgement
            j = row[:memory_judgement]
            if j === :improvement
                return isgood
            elseif j === :regression
                return !isgood
            end
        end

        return false
    end

    return highlighter_judgement
end

function make_formatter(data::Vector{<:TrialJudgementRow})
    names = Tables.columnnames(eltype(data))
    function formatter_judgement(x, i, j)
        name = names[j]

        if name in (:time, :memory, :allocs, :gctime)
            return BenchmarkTools.prettydiff(x)
        elseif name in (:time_tolerance, :memory_tolerance)
            return BenchmarkTools.prettypercent(x)
        end

        return x
    end

    return formatter_judgement
end

function Base.show(io::IO, ::MIME"text/plain", rows::Vector{<:TrialJudgementRow})
    hgood = Highlighter(make_highlighter_judgement(true); foreground=:green, bold=true)
    hbad = Highlighter(make_highlighter_judgement(false); foreground=:red, bold=true)
    formatter = make_formatter(rows)
    return pretty_table(io, rows; highlighters=(hgood, hbad), formatters=(formatter,))
end

function Base.show(io::IO, ::MIME"text/html", rows::Vector{<:TrialJudgementRow})
    hgood = HTMLHighlighter(
        make_highlighter_judgement(true),
        HTMLDecoration(; color="green", font_weight="bold"),
    )
    hbad = HTMLHighlighter(
        make_highlighter_judgement(false), HTMLDecoration(; color="red", font_weight="bold")
    )
    formatter = make_formatter(rows)
    return pretty_table(
        io,
        rows;
        backend=Val(:html),
        highlighters=(hgood, hbad),
        formatters=(formatter,),
        tf=PrettyTables.tf_html_minimalist,
    )
end

#########################################################
# Make it more convenient to load benchmarks into table #
#########################################################
function judgementtable(
    results::AbstractVector,
    results_old::AbstractVector,
    extras=fill(NamedTuple(), length(results));
    stat=minimum,
)
    @assert length(results_old) == length(results) "benchmarks have different lengths"

    return collect(
        TrialJudgementRow(
            groupname,
            judge(stat(results[i][groupname]), stat(results_old[i][groupname])),
            extras[i],
        ) for i in eachindex(results) for
        groupname in keys(results[i]) if groupname in keys(results_old[i])
    )
end

function judgementtable(name::String, name_old::String; kwargs...)
    model_names =
        map(filter(endswith("_benchmarks.json"), readdir(projectdir("results", name)))) do x
            # Strip the suffix.
            x[1:(end - 5)]
        end

    results = []
    results_old = []
    for model_name in model_names
        append!(
            results, BenchmarkTools.load(projectdir("results", name, "$(model_name).json"))
        )
        append!(
            results_old,
            BenchmarkTools.load(projectdir("results", name_old, "$(model_name).json")),
        )
    end

    extras = [(model_name=model_name,) for model_name in model_names]

    return judgementtable(results, results_old, extras; kwargs...)
end

function judgementtable_single(
    results::AbstractVector,
    reference_group::AbstractString,
    extras=fill(NamedTuple(), length(results));
    stat=minimum,
)
    return collect(
        TrialJudgementRow(
            groupname,
            judge(stat(results[i][groupname]), stat(results[i][reference_group])),
            extras[i],
        ) for i in eachindex(results) for groupname in keys(results[i])
    )
end

function judgementtable_single(
    name::AbstractString, reference_group::AbstractString; kwargs...
)
    model_names =
        map(filter(endswith("_benchmarks.json"), readdir(projectdir("results", name)))) do x
            # Strip the suffix.
            x[1:(end - 5)]
        end

    results = []
    for model_name in model_names
        append!(
            results, BenchmarkTools.load(projectdir("results", name, "$(model_name).json"))
        )
    end

    extras = [(model_name=model_name,) for model_name in model_names]

    return judgementtable_single(results, reference_group, extras; kwargs...)
end
