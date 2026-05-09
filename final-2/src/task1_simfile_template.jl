"""
Template simulation file for run_task1_flowunsteady.jl

You can copy this and replace internals with your case setup.
"""

using FLOWUnsteady
using FLOWVPM
using FLOWVLM

# Include your existing case files here, for example:
# include("/abs/path/to/vehicle_definition_wing.jl")
# include("/abs/path/to/maneuver_definition_wing.jl")
# include("/abs/path/to/simulation_definition_wing.jl")

"""
Must return:
- sim: FLOWUnsteady simulation object
- run_kwargs: Dict with at least :nsteps and any run_simulation kwargs
"""
function build_task1_simulation()
    # -------------------------------------------------------
    # Replace this block with your actual simulation creation.
    # -------------------------------------------------------
    # sim = ...
    # nsteps = ...
    # run_kwargs = Dict{Symbol,Any}(
    #     :nsteps => nsteps,
    #     :Vinf => Vinf,
    #     :rho => rho,
    #     :mu => mu,
    # )

    error("Edit task1_simfile_template.jl and implement build_task1_simulation()")
end
