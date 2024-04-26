"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    use_contact_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.use_contact_shadow = use_contact_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    if light_type == "SUN":
        light_data.angle = 1.57
    return light_object


def randomize_lighting():
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    # Add random angle offset in 0-90
    angle_offset = random.uniform(0, math.pi / 2)
    

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398 + angle_offset), # 45 0 -45
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699 + angle_offset), # -45 0 -225
        energy=random.choice([2.5, 3.25, 4]),
        use_shadow=True,
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619 + angle_offset), # 45 0 135
        energy=random.choice([2, 3, 3.5]),
    )

    # Create small light
    small_light1 = _create_light(
        name="S1_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 0.785398 + angle_offset), # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    small_light2 = _create_light(
        name="S2_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(1.57079, 0, 3.92699 + angle_offset), # 90 0 45
        energy=random.choice([0.25, 0.5, 1]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0), # 180 0 0
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
        small_light1=small_light1,
        small_light2=small_light2,
    )

def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 300000
    bpy.data.objects["Area"].location[2] = 10
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100

def normal_file_output():
    # create input render layer node
    render_layers = scene.node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.label = "Custom Outputs"
    render_layers.name = "Custom Outputs"

    # create normal output node
    normal_file_output = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "normal"
    normal_file_output.name = "normal"
    normal_file_output.base_path = ""

    # Create a Separate RGB node
    separate_rgb_node = scene.node_tree.nodes.new(type="CompositorNodeSepRGBA")
    scene.node_tree.links.new(
        render_layers.outputs["Normal"], separate_rgb_node.inputs["Image"]
    )

    # Create a Combine RGBA node
    combine_rgba_node = scene.node_tree.nodes.new(type="CompositorNodeCombRGBA")
    reversed_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
    reversed_G.operation = "MULTIPLY"
    reversed_G.inputs[0].default_value = -1

    # y -> -y
    scene.node_tree.links.new(separate_rgb_node.outputs["G"], reversed_G.inputs[1])

    # map normal range from [-1, 1] to [0, 1]
    # channel R
    bias_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
    bias_node_R.operation = "ADD"
    bias_node_R.inputs[0].default_value = 1

    scale_node_R = scene.node_tree.nodes.new(type="CompositorNodeMath")
    scale_node_R.operation = "MULTIPLY"
    scale_node_R.inputs[0].default_value = 0.5

    scene.node_tree.links.new(separate_rgb_node.outputs["R"], bias_node_R.inputs[1])
    scene.node_tree.links.new(bias_node_R.outputs[0], scale_node_R.inputs[1])

    # channel G
    bias_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
    bias_node_G.operation = "ADD"
    bias_node_G.inputs[0].default_value = 1

    scale_node_G = scene.node_tree.nodes.new(type="CompositorNodeMath")
    scale_node_G.operation = "MULTIPLY"
    scale_node_G.inputs[0].default_value = 0.5

    scene.node_tree.links.new(reversed_G.outputs[0], bias_node_G.inputs[1])
    scene.node_tree.links.new(bias_node_G.outputs[0], scale_node_G.inputs[1])

    # channel B
    bias_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
    bias_node_B.operation = "ADD"
    bias_node_B.inputs[0].default_value = 1

    scale_node_B = scene.node_tree.nodes.new(type="CompositorNodeMath")
    scale_node_B.operation = "MULTIPLY"
    scale_node_B.inputs[0].default_value = 0.5

    scene.node_tree.links.new(separate_rgb_node.outputs["B"], bias_node_B.inputs[1])
    scene.node_tree.links.new(bias_node_B.outputs[0], scale_node_B.inputs[1])

    # Combine RGB
    scene.node_tree.links.new(
        combine_rgba_node.inputs["R"], scale_node_R.outputs[0]
    )
    scene.node_tree.links.new(
        combine_rgba_node.inputs["G"], scale_node_B.outputs[0]
    )
    scene.node_tree.links.new(
        combine_rgba_node.inputs["B"], scale_node_G.outputs[0]
    )
    scene.node_tree.links.new(
        combine_rgba_node.inputs["A"], separate_rgb_node.outputs["A"]
    )

    # add alpha channel
    set_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
    set_alpha_node.mode = "REPLACE_ALPHA"
    scene.node_tree.links.new(
        combine_rgba_node.outputs["Image"], set_alpha_node.inputs["Image"]
    )
    scene.node_tree.links.new(
        render_layers.outputs["Alpha"], set_alpha_node.inputs["Alpha"]
    )

    scene.node_tree.links.new(
        set_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
    )


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
    group_name: str,
    normal: str,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # normalize the scene
    normalize_scene()

    # randomize the lighting
    randomize_lighting()
    # add_lighting()

    object_uid = os.path.basename(object_file).split(".")[0]

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # ortho
    if args.ortho_camera:
        print("use ortho camera")
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = args.ortho_scale

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    # metadata_path = os.path.join(output_dir, "metadata.json")
    # os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    # with open(metadata_path, "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, sort_keys=True, indent=2)

    if normal:
        scene.use_nodes = True
        context.view_layer.use_pass_normal = True  # for normal rendering
        scene.view_settings.view_transform = "Raw"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        normal_file_output()

    # render the images
    theta_list_2 = [0.5, 0.75]
    theta_list_5 = [0.375, 0.5, 0.625, 0.75, 0.875]
    theta_list_7 = [0.33, 0.41, 0.5, 0.58, 0.67, 0.75, 0.83]
    for i in range(num_renders):
        # set camera
        # camera = randomize_camera(
        #     only_northern_hemisphere=only_northern_hemisphere,
        # )
        # set the camera position
        if num_renders == 2:
            theta = theta_list_2[i] * math.pi * 2
        elif num_renders == 5:
            theta = theta_list_5[i] * math.pi * 2
        elif num_renders == 7:
            theta = theta_list_7[i] * math.pi * 2
        else:
            theta = (i / args.num_renders) * math.pi * 2

        if args.elevation != "rand":
            phi = math.radians(90 - float(args.elevation))
        else:
            phi = math.radians(90 - random.randint(-5, 30))


        camera_dist = args.camera_dist
        print(camera_dist)
        point = (
            camera_dist * math.sin(phi) * math.cos(theta),
            camera_dist * math.sin(phi) * math.sin(theta),
            camera_dist * math.cos(phi),
        )
        cam.location = point

        if normal:
            prefix = "normals_"
        else:
            prefix = ""
    
        object_uid = "_".join(object_uid.lower().split(" "))
        render_path = os.path.join(
            output_dir,
            object_uid,
            f"{prefix}{i:03d}.png",
        )
        
        scene.render.filepath = render_path


        if normal:
            render_path = os.path.join(
                output_dir,
                object_uid,
                f"{prefix}{i:03d}_",
            )
            scene.node_tree.nodes['normal'].file_slots[0].path = render_path
            scene.render.filepath = './.tmp/tmp.png'

        
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        required=False,
        default=None,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--elevation",
        type=str,
        required=True,
        default="rand",
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--use_contact_shadow",
        action="store_true",
        help="",
        default=False,
    )
    parser.add_argument(
        "--use_shadow",
        action="store_true",
        help="",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--camera_dist",
        type=float,
        default=1.5,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--camera_dist_min",
        type=float,
        default=1.5,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--camera_dist_max",
        type=float,
        default=1.5,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--ortho_camera",
        action="store_true",
        help="use ortho camera and ortho scale",
        default=False,
    )
    parser.add_argument(
        "--ortho_scale",
        type=float,
        default=1.2,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--normal",
        type=str,
        default=None,
        help="Number of renders to save of the object.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "CUDA", "OPENCL"

    # Render the images
    render_object(
            object_file=args.object_path,
            num_renders=args.num_renders,
            only_northern_hemisphere=args.only_northern_hemisphere,
            output_dir=args.output_dir,
            group_name=args.group_name,
            normal=args.normal,
        )
