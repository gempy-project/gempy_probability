from enum import Enum, auto
import gempy as gp
import os
from gempy_engine.core.backend_tensor import BackendTensor


class ExampleProbModel(Enum):
    TWO_WELLS = auto()


def generate_example_model(example_model: ExampleProbModel, compute_model: bool = True) -> gp.data.GeoModel:
    match example_model:
        case ExampleProbModel.TWO_WELLS:
            return _generate_two_wells_model(compute_model)
        case _:
            raise ValueError(f"Example model {example_model} not found.")


def _generate_two_wells_model(compute_model: bool) -> gp.data.GeoModel:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'examples', 'tutorials', 'data'))
    geo_model = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        refinement=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=os.path.join(data_path, "2-layers", "2-layers_orientations.csv"),
            path_to_surface_points=os.path.join(data_path, "2-layers", "2-layers_surface_points.csv")
        )
    )

    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False

    if compute_model:
        # Compute the geological model
        gp.compute_model(geo_model)

    return geo_model
