import Literate

foreach(["par_ldiv.jl", "iternz.jl"]) do i
    Literate.markdown(joinpath("lit", i), "src/"; execute=true, repo_root_path="../")
end