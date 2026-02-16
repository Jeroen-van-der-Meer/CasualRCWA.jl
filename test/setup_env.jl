using Pkg
test_dir = @__DIR__
cd(test_dir)
Pkg.activate(".")
Pkg.develop(path=joinpath(test_dir, ".."))