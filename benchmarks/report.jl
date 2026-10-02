# Usage: julia benchmarks/report.jl REPO PR_SHA MAIN_SHA PR_TABLE [MAIN_TABLE]
# Uses only Base so post-comment does not need the benchmark environment.
function read_table(path)
    lines = strip.(readlines(path))
    if length(lines) >= 2 && first(lines) == last(lines) == "```"
        lines = lines[2:(end - 1)]
    end
    malformed() = error("Malformed benchmark table: $path")
    length(lines) >= 7 || malformed()
    columns(line) = split(line, r" {2,}")
    header = [
        "Model", "dim", "linked", "primal", "FwdDiff", "RvsDiff", "Mooncake", "Enzyme"
    ]
    occursin(r"^=+$", lines[1]) && lines[end] == lines[1] || malformed()
    columns(lines[2]) == ["eval", "gradient"] || malformed()
    occursin(r"^-+  -+$", lines[3]) || malformed()
    columns(lines[4]) == header || malformed()
    lines[5] == repeat("-", length(lines[1])) || malformed()

    seen = Set{Tuple{String,String,String}}()
    return map(lines[6:(end - 1)]) do line
        # print_results uses a two-space gap; model names contain only single
        # spaces, as does the separator between a primal number and its unit.
        cells = columns(line)
        length(cells) == length(header) || malformed()
        name, dim, linked, primal = cells[1:4]
        isempty(name) && malformed()
        occursin(r"^(?:[0-9]+|err)$", dim) || malformed()
        linked in ("true", "false") || malformed()
        if primal != "err"
            time = split(primal, ' ')
            length(time) == 2 || malformed()
            tryparse(Float64, time[1]) !== nothing || malformed()
            time[2] in ("ns", "μs", "ms", "s") || malformed()
        end
        ratios = cells[5:8]
        all(x -> x == "err" || tryparse(Float64, x) !== nothing, ratios) || malformed()
        # The tiny-primal marker can differ between runs; it is not part of
        # the model's identity. Preserve the PR marker in the displayed name.
        key = (chopsuffix(name, "*"), dim, linked)
        key in seen && malformed()
        push!(seen, key)
        (; key, name, dim, linked, primal, ratios)
    end
end

function commit_header(repo, sha, label)
    isempty(sha) && return "`$label`"
    return "`$label` [$(first(sha, 7))](https://github.com/$repo/commit/$sha)"
end

function report(args)
    length(args) in (4, 5) || error(
        "Usage: julia benchmarks/report.jl REPO PR_SHA MAIN_SHA PR_TABLE [MAIN_TABLE]"
    )
    repo, pr_sha, main_sha, pr_path = args[1:4]
    pr_rows = read_table(pr_path)
    main_times = Dict{Tuple{String,String,String},String}()
    main_note = ""
    if length(args) == 5
        # Main runs main's printer, which may predate a format change in this PR.
        main_rows = try
            read_table(args[5])
        catch err
            err isa ErrorException || rethrow()
            @warn "Could not parse the main benchmark table" exception = err
            main_note = "Main benchmark table could not be parsed; see workflow logs."
            ()
        end
        for row in main_rows
            main_times[row.key] = row.primal
        end
    end
    rows = map(pr_rows) do row
        (
            row.name,
            row.dim,
            row.linked,
            get(main_times, row.key, "—"),
            row.primal,
            row.ratios...,
        )
    end
    header = (
        "Model", "dim", "linked", "main", "PR", "FwdDiff", "RvsDiff", "Mooncake", "Enzyme"
    )
    # Match print_results: one extra space for names, two for every other
    # column, and a two-space gap between columns.
    widths = [
        max(length(label), maximum(textwidth(row[i]) for row in rows)) + (i == 1 ? 1 : 2)
        for (i, label) in enumerate(header)
    ]
    gap = "  "
    gap_w = textwidth(gap)
    stub_w = sum(widths[1:3]) + 2 * gap_w
    primal_w = sum(widths[4:5]) + gap_w
    grad_w = sum(widths[6:9]) + 3 * gap_w
    total_w = stub_w + gap_w + primal_w + gap_w + grad_w
    center(s, w) = lpad(rpad(s, div(w + textwidth(s), 2)), w)
    function format_row(row)
        return join(
            (
                i == 1 ? rpad(cell, w) : lpad(cell, w) for
                (i, (cell, w)) in enumerate(zip(row, widths))
            ),
            gap,
        )
    end

    println(
        "Primal times: $(commit_header(repo, main_sha, "main")), $(commit_header(repo, pr_sha, "PR"))",
    )
    println("\n```")
    println(repeat("=", total_w))
    println(
        rpad("", stub_w) *
        gap *
        center("primal", primal_w) *
        gap *
        center("gradient", grad_w),
    )
    println(rpad("", stub_w) * gap * repeat("-", primal_w) * gap * repeat("-", grad_w))
    println(format_row(header))
    println(repeat("-", total_w))
    foreach(row -> println(format_row(row)), rows)
    println(repeat("=", total_w))
    println("```")
    isempty(main_note) || println("\n", main_note)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    report(ARGS)
end
