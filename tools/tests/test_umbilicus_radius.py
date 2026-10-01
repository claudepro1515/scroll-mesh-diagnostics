import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from umbilicus_radius import umbilicus_xy_at_z, radius_stats_from_arrays  # noqa: E402


def test_umbilicus_xy_at_z_interpolates_linearly():
    # eje recto en x=100+z*0.1, y=200 fijo, dos puntos de control bastan para probar interpolacion
    cp = [{"x": 100.0, "y": 200.0, "z": 0.0}, {"x": 200.0, "y": 200.0, "z": 1000.0}]
    cx, cy = umbilicus_xy_at_z(cp, np.array([0.0, 500.0, 1000.0]))
    assert np.allclose(cx, [100.0, 150.0, 200.0])
    assert np.allclose(cy, [200.0, 200.0, 200.0])


def test_umbilicus_xy_at_z_sorts_unordered_control_points():
    # mismos puntos que arriba pero en orden invertido en la lista -- debe dar lo mismo
    cp = [{"x": 200.0, "y": 200.0, "z": 1000.0}, {"x": 100.0, "y": 200.0, "z": 0.0}]
    cx, cy = umbilicus_xy_at_z(cp, np.array([500.0]))
    assert np.allclose(cx, [150.0])


def test_radius_stats_same_voxel_size_ring_of_points():
    # anillo de puntos exactos a radio 50 alrededor de un eje recto en (0,0), mismas unidades
    # (mesh_voxel_um == umb_voxel_um): el radio medido debe dar exactamente 50 en todos los percentiles
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x = (50 * np.cos(theta)).reshape(1, -1)
    y = (50 * np.sin(theta)).reshape(1, -1)
    z = np.full_like(x, 100.0)
    valid = np.ones_like(x, dtype=bool)
    cp = [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 1000.0}]
    stats = radius_stats_from_arrays(x, y, z, valid, cp, umb_voxel_um=9.362, mesh_voxel_um=9.362)
    assert np.isclose(stats["radius_median"], 50.0, atol=1e-6)
    assert np.isclose(stats["radius_p10"], 50.0, atol=1e-6)
    assert stats["n_valid"] == 64


def test_radius_stats_converts_voxel_size_mismatch():
    # el caso real (PHerc0139): umbilicus a 2.399um, malla a 9.362um. Si el umbilicus estuviera en
    # unidades de malla (9.362um) el eje pasaria por (0,0); publicado a 2.399um, el mismo eje fisico
    # se escribe en control_points como (0,0) * (9.362/2.399) = (0,0) igual (eje en el origen no
    # revela el bug de escala) -- se desplaza el eje para que un error de conversion sí cambie el radio.
    mesh_voxel_um = 9.362
    umb_voxel_um = 2.399
    # eje fisico real en x=50 micras (constante), a lo largo de z. En unidades de malla (9.362um)
    # eso es x = 50/9.362 = 5.341 voxeles-malla. Publicado en el umbilicus.json a 2.399um: mismo
    # x fisico (50 um) / 2.399 = 20.84 voxeles-umbilicus.
    x_axis_umb_voxels = 50.0 / umb_voxel_um
    cp = [{"x": x_axis_umb_voxels, "y": 0.0, "z": 0.0},
          {"x": x_axis_umb_voxels, "y": 0.0, "z": 100000.0}]
    # anillo de la malla a radio fisico real 200um alrededor de ese eje, en unidades de malla
    x_axis_mesh_voxels = 50.0 / mesh_voxel_um
    radius_mesh_voxels = 200.0 / mesh_voxel_um
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x = (x_axis_mesh_voxels + radius_mesh_voxels * np.cos(theta)).reshape(1, -1)
    y = (radius_mesh_voxels * np.sin(theta)).reshape(1, -1)
    z = np.full_like(x, 500.0)  # voxeles de malla, dentro del rango de z del umbilicus convertido
    valid = np.ones_like(x, dtype=bool)
    stats = radius_stats_from_arrays(x, y, z, valid, cp, umb_voxel_um=umb_voxel_um, mesh_voxel_um=mesh_voxel_um)
    assert np.isclose(stats["radius_median"], radius_mesh_voxels, atol=1e-3)


def test_radius_stats_wrong_voxel_size_gives_wrong_answer():
    # mismo caso que arriba, pero SIN la conversion (asumiendo umb_voxel_um == mesh_voxel_um por
    # error): confirma que el bug de escala (el que casi se comete en la corrida real) sí se nota
    # como un radio muy distinto -- este test documenta por que la conversion es necesaria, no solo
    # que la funcion "funciona".
    mesh_voxel_um = 9.362
    umb_voxel_um = 2.399
    x_axis_umb_voxels = 50.0 / umb_voxel_um
    cp = [{"x": x_axis_umb_voxels, "y": 0.0, "z": 0.0},
          {"x": x_axis_umb_voxels, "y": 0.0, "z": 100000.0}]
    x_axis_mesh_voxels = 50.0 / mesh_voxel_um
    radius_mesh_voxels = 200.0 / mesh_voxel_um
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    x = (x_axis_mesh_voxels + radius_mesh_voxels * np.cos(theta)).reshape(1, -1)
    y = (radius_mesh_voxels * np.sin(theta)).reshape(1, -1)
    z = np.full_like(x, 500.0)
    valid = np.ones_like(x, dtype=bool)
    wrong = radius_stats_from_arrays(x, y, z, valid, cp, umb_voxel_um=mesh_voxel_um, mesh_voxel_um=mesh_voxel_um)
    assert not np.isclose(wrong["radius_median"], radius_mesh_voxels, atol=1.0)
