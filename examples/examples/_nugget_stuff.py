import gempy as gp
from vector_geology.model_building_functions import optimize_nuggets_for_group
import gempy as gp
import numpy as np


def optimize_nuggets_for_group(geo_model: gp.data.GeoModel, structural_group: gp.data.StructuralGroup,
                               plot_evaluation: bool = False, plot_result: bool = False) -> None:
    temp_structural_frame = gp.data.StructuralFrame(
        structural_groups=[structural_group],
        color_gen=gp.data.ColorsGenerator()
    )

    previous_structural_frame = geo_model.structural_frame

    geo_model.structural_frame = temp_structural_frame

    gp.API.compute_API.optimize_and_compute(
        geo_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH,
        ),
        max_epochs=100,
        convergence_criteria=1e5
    )

    nugget_effect = geo_model.taped_interpolation_input.surface_points.nugget_effect_scalar.detach().numpy()
    np.save(f"temp/nuggets_{structural_group.name}", nugget_effect)

    if plot_evaluation:
        import matplotlib.pyplot as plt

        plt.hist(nugget_effect, bins=50, color='black', alpha=0.7, log=True)
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Histogram of Eigenvalues (nugget-grad)')
        plt.show()

    if plot_result:
        import gempy_viewer as gpv
        import pyvista as pv

        gempy_vista = gpv.plot_3d(
            model=geo_model,
            show=False,
            kwargs_plot_structured_grid={'opacity': 0.3}
        )

        # Create a point cloud mesh
        surface_points_xyz = geo_model.surface_points_copy.df[['X', 'Y', 'Z']].to_numpy()

        point_cloud = pv.PolyData(surface_points_xyz[0:])
        point_cloud['values'] = nugget_effect

        gempy_vista.p.add_mesh(
            point_cloud,
            scalars='values',
            cmap='inferno',
            point_size=25,
        )

        gempy_vista.p.show()

    geo_model.structural_frame = previous_structural_frame
    return nugget_effect


def optimize_nuggets_for_whole_project(geo_model: gp.data.GeoModel):
    geo_model.interpolation_options.kernel_options.range = 0.7
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=geo_model.structural_frame.get_group_by_name('Red'),
        plot_evaluation=False,
        plot_result=True
    )
    geo_model.interpolation_options.kernel_options.range = 2
    geo_model.interpolation_options.kernel_options.c_o = 4
    optimize_nuggets_for_group(
        geo_model=geo_model,
        structural_group=geo_model.structural_frame.get_group_by_name('Blue'),
        plot_evaluation=False,
        plot_result=False
    )
    if False:
        optimize_nuggets_for_group(
            geo_model=geo_model,
            structural_group=geo_model.structural_frame.get_group_by_name('Green'),
            plot_evaluation=False,
            plot_result=True
        )


def apply_optimized_nuggets(geo_model: gp.data.GeoModel, loaded_nuggets_red, loaded_nuggets_blue, loaded_nuggets_green):
    gp.modify_surface_points(
        geo_model,
        slice=None,
        elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Red').elements],
        nugget=loaded_nuggets_red
    )
    if True:  # Ignore OB
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Blue').elements],
            nugget=loaded_nuggets_blue
        )
    if False:
        gp.modify_surface_points(
            geo_model,
            slice=None,
            elements_names=[element.name for element in geo_model.structural_frame.get_group_by_name('Green').elements],
            nugget=loaded_nuggets_green
        )

