"""World-space support bounds of the actual collision shapes."""

import mujoco
import numpy as np


def geom_bounds(model, data, geom):
    center = data.geom_xpos[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        world = vertices @ rotation.T + center
        return world.min(axis=0), world.max(axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * np.abs(rotation[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        extent = size[0] * np.sqrt(np.maximum(0, 1 - axis**2)) + size[1] * np.abs(axis)
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.sqrt(np.sum((rotation * size) ** 2, axis=1))
    else:
        raise ValueError(f"Unsupported object collision shape: {kind}")
    return center - extent, center + extent


def object_bounds(model, data, geoms):
    bounds = [geom_bounds(model, data, geom) for geom in geoms]
    return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)
