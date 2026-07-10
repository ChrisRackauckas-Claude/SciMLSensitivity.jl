using SciMLSensitivity
using Test

function exported_names(mod)
    return Set(
        filter!(
            name -> name != nameof(mod),
            collect(names(mod; all = false, imported = false))
        )
    )
end

function documented_names(root)
    docs = Set{Symbol}()
    for (dir, _, files) in walkdir(joinpath(root, "docs", "src"))
        for file in files
            endswith(file, ".md") || continue
            in_docs_block = false
            for raw_line in eachline(joinpath(dir, file))
                line = strip(raw_line)
                if startswith(line, "```@docs")
                    in_docs_block = true
                    continue
                elseif in_docs_block && startswith(line, "```")
                    in_docs_block = false
                    continue
                end
                in_docs_block || continue
                isempty(line) && continue
                startswith(line, "#") && continue

                name = first(split(line))
                name = replace(name, "SciMLSensitivity." => "")
                push!(docs, Symbol(name))
            end
        end
    end
    return docs
end

function has_docstring(mod, name)
    object = getfield(mod, name)
    doc = sprint(show, MIME"text/plain"(), Docs.doc(object))
    return !contains(doc, "No documentation found")
end

root = normpath(joinpath(@__DIR__, "..", ".."))
exports = exported_names(SciMLSensitivity)
rendered_docs = documented_names(root)

@testset "exported API has docstrings" begin
    missing_docstrings = sort!(collect(filter(name -> !has_docstring(SciMLSensitivity, name), exports)))
    @test isempty(missing_docstrings)
end

@testset "exported API is rendered in docs" begin
    missing_rendered_docs = sort!(collect(setdiff(exports, rendered_docs)))
    @test isempty(missing_rendered_docs)
end
