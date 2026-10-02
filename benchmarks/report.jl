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

function escape_html(s)
    return replace(
        s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;"
    )
end

function commit_header(repo, sha, label)
    isempty(sha) && return label
    url = escape_html("https://github.com/$repo/commit/$sha")
    return "<a href=\"$url\">$label ($(escape_html(first(sha, 7))))</a>"
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
    println("<table>")
    println("<thead>")
    println(
        "<tr><th rowspan=\"2\">Model</th><th rowspan=\"2\">dim</th>" *
        "<th rowspan=\"2\">linked</th><th colspan=\"2\">primal</th>" *
        "<th colspan=\"4\">gradient</th></tr>",
    )
    println(
        "<tr><th>$(commit_header(repo, main_sha, "main"))</th>" *
        "<th>$(commit_header(repo, pr_sha, "PR"))</th>" *
        "<th>FwdDiff</th><th>RvsDiff</th><th>Mooncake</th><th>Enzyme</th></tr>",
    )
    println("</thead>")
    println("<tbody>")
    for row in pr_rows
        cells = (
            row.name,
            row.dim,
            row.linked,
            get(main_times, row.key, "—"),
            row.primal,
            row.ratios...,
        )
        println("<tr>", join(("<td>$(escape_html(cell))</td>" for cell in cells)), "</tr>")
    end
    println("</tbody>")
    println("</table>")
    isempty(main_note) || println("\n", main_note)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    report(ARGS)
end
