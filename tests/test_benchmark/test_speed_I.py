import pytest

import gempy as gp
import gempy_viewer as gpv
import numpy as np

from gempy_probability.plugins.examples_generator import generate_example_model, ExampleProbModel


def test_speed_I():
    
    two_wells: gp.data.GeoModel = generate_example_model(
        example_model=ExampleProbModel.TWO_WELLS,
        compute_model=False
    )

    assert two_wells.interpolation_options.number_octree_levels == 4, "Number of octrees should be 4"

    # region Minimal grid for the specific likelihood function
    x_loc = 6000
    y_loc = 0
    z_loc = np.linspace(0, 4000, 100)
    xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])
    gp.set_custom_grid(two_wells.grid, xyz_coord=xyz_coord)
    # endregion

    two_wells.grid.active_grids = gp.data.Grid.GridTypes.CUSTOM
    
    profiler = cProfile.Profile()
    profiler.enable()
    iterations = 100
    for _ in range(iterations):
        gp.compute_model(
            gempy_model=two_wells,
            engine_config=gp.data.GemPyEngineConfig(
                backend=gp.data.AvailableBackends.numpy
            )
        )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)

    if PLOT := False:
        gpv.plot_2d(two_wells, show_scalar=False)


@pytest.mark.skip(reason="Not implemented yet")
def test_speed_on_gravity_likelihood():
    pass