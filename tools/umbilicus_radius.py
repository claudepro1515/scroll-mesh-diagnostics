"""Frente G, paso 2c: radio real (distancia al eje del rollo = umbilicus) de una malla tifxyz.

Por que hace falta: wm/mesh_topology_check.py (neighbor_gap_report / check_neighbor_gap_pair)
detecto que dos vueltas de comunidad (armando-gaona, PHerc0211, w010-w011/w011-w012) dan mas
%-anomalo (3.77/5.37 %) que la linea base de control de PHerc0139 (0.08-0.60 % tipico, cola hasta
3.36 %) -- pero el numero de vuelta wNNN NO es comparable entre rollos DISTINTOS (rollos de
tamano/enrollado distinto numeran sus vueltas de forma independiente), y las vueltas mas internas
de CUALQUIER rollo se comprimen mas de forma fisica normal. Este modulo mide la distancia real (en
voxeles fisicos) de una malla al eje del rollo (umbilicus), para poder comparar "que tan interna"
es una vuelta entre rollos distintos con una unidad fisica real, no con el indice wNNN.

OJO con las unidades: el umbilicus.json de un rollo puede estar publicado sobre un volumen de OTRA
resolucion que la malla que se quiere medir (visto en PHerc0139: umbilicus a 2.399 um, mallas
curadas de segments/ a 9.362 um -- confirmado leyendo metadata.voxelsize_um de cada umbilicus.json
y comparando contra el nombre de archivo de la malla, p.ej. "...-9.362um.tifxyz"). Hay que pasar
ambos a MICRAS FISICAS antes de interpolar/comparar, y solo al final convertir el resultado a las
unidades de voxel de la malla para que el radio quede en las mismas unidades que el resto de
wm/mesh_topology_check.py.

Uso:
  python wm/umbilicus_radius.py <mesh_dir_con_x.tif_y.tif_z.tif> <umbilicus.json> [--mesh-voxel-um 9.362]
Imprime una linea JSON: {n_valid, z_min, z_max, radius_median, radius_mean, radius_p10, radius_p90}
(radio en voxeles de la malla, mesh_voxel_um por defecto).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_topology_check import load_tifxyz  # noqa: E402


def umbilicus_xy_at_z(control_points: list[dict], z_umb_units: np.ndarray):
    """Interpola (x,y) del umbilicus en z_umb_units (mismas unidades de voxel que control_points).
    Los control_points no siempre vienen ordenados por z en el JSON publicado -- se ordenan aqui."""
    zs = np.array([p["z"] for p in control_points], dtype=np.float64)
    xs = np.array([p["x"] for p in control_points], dtype=np.float64)
    ys = np.array([p["y"] for p in control_points], dtype=np.float64)
    order = np.argsort(zs)
    zs, xs, ys = zs[order], xs[order], ys[order]
    cx = np.interp(z_umb_units, zs, xs)
    cy = np.interp(z_umb_units, zs, ys)
    return cx, cy


def radius_stats_from_arrays(x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray,
                              control_points: list[dict], umb_voxel_um: float,
                              mesh_voxel_um: float = 9.362) -> dict:
    """Logica pura numpy (offline-testable). control_points y umb_voxel_um vienen de un
    umbilicus.json ya cargado (ver umbilicus_radius_report para el caso de uso real con I/O)."""
    zv, xv, yv = z[valid], x[valid], y[valid]
    z_in_umb_units = zv * (mesh_voxel_um / umb_voxel_um)
    cx_umb, cy_umb = umbilicus_xy_at_z(control_points, z_in_umb_units)
    cx = cx_umb * (umb_voxel_um / mesh_voxel_um)
    cy = cy_umb * (umb_voxel_um / mesh_voxel_um)
    r = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
    return {
        "n_valid": int(valid.sum()),
        "z_min": float(zv.min()) if zv.size else None,
        "z_max": float(zv.max()) if zv.size else None,
        "radius_median": float(np.median(r)),
        "radius_mean": float(r.mean()),
        "radius_p10": float(np.percentile(r, 10)),
        "radius_p90": float(np.percentile(r, 90)),
    }


def umbilicus_radius_report(mesh_dir: str, umbilicus_path: str, mesh_voxel_um: float = 9.362) -> dict:
    x, y, z, valid = load_tifxyz(mesh_dir)
    umb_doc = json.load(open(umbilicus_path))
    umb_voxel_um = umb_doc["metadata"]["voxelsize_um"]
    stats = radius_stats_from_arrays(x, y, z, valid, umb_doc["control_points"], umb_voxel_um, mesh_voxel_um)
    stats["mesh_dir"] = mesh_dir
    stats["umbilicus_path"] = umbilicus_path
    stats["umb_voxel_um"] = umb_voxel_um
    stats["mesh_voxel_um"] = mesh_voxel_um
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_dir")
    ap.add_argument("umbilicus_json")
    ap.add_argument("--mesh-voxel-um", type=float, default=9.362)
    a = ap.parse_args()
    print(json.dumps(umbilicus_radius_report(a.mesh_dir, a.umbilicus_json, a.mesh_voxel_um)))
