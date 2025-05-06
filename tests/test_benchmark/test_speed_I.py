import gempy as gp
import gempy_viewer as gpv
from gempy_probability.plugins.examples_generator import generate_example_model, ExampleProbModel


def test_speed_I():
    two_wells: gp.data.GeoModel = generate_example_model(
        example_model=ExampleProbModel.TWO_WELLS,
        compute_model=False
    )

    assert two_wells.interpolation_options.number_octree_levels == 4, "Number of octrees should be 4"

    gp.compute_model(two_wells)

    if PLOT := False:
        gpv.plot_2d(two_wells, show_scalar=False)
