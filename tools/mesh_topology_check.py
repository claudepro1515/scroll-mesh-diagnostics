"""Frente G (herramientas de diagnostico reutilizables): detector de pliegues/autointersecciones
en una malla tifxyz de una sola vuelta (wNNN, como las escribe fit_spiral).

Idea: cada wNNN es UNA hoja (una vuelta) parametrizada en una rejilla (fila, columna). Si el ajuste
es correcto, la fila (aprox. el eje Z del rollo, ver release/tools/winding_agreement.py:
load_triangles filtra filas por rango de Z) avanza suavemente en 3D al recorrer las columnas: la
hoja no deberia doblarse sobre si misma dentro de una misma fila. Un giro brusco (>90 grados) entre
el paso de entrada y el paso de salida de un vertice es una senal objetiva de que la malla se dobla
o se autointerseca localmente -- un defecto real de "topologia"/"reparacion de mallas" (problemas
abiertos oficiales del reto), no una curva suave del papiro.

Esto es 100% offline una vez que se tienen los x.tif/y.tif/z.tif (no hace falta CT ni umbilicus):
mas barato que sheet_hits.py (que necesita bajar el volumen crudo por rayos) y complementario --
sheet_hits mide si la hoja SIGUE al papiro; esto mide si la malla es geometricamente consistente
consigo misma.

Metodo (row_fold_report):
  Para cada fila valida, se recorren sus columnas validas EN ORDEN y se calculan los vectores paso
  entre vertices consecutivos (en 3D, voxeles). En cada vertice interior con paso de entrada e_in y
  paso de salida e_out (ambos de longitud >= min_step, para no medir ruido en pasos casi nulos), se
  calcula el coseno del angulo entre -e_in y e_out: 1.0 = sigue derecho, 0.0 = giro de 90 grados,
  -1.0 = se devuelve exactamente por donde vino (pliegue perfecto). Un vertice con
  cos(angulo) < --min-cos (por defecto -0.3, ~107 grados) se marca como pliegue.

Verificado con datos sinteticos (wm/tests/test_mesh_topology_check.py) ANTES de usarse con datos
reales: un arco suave (cuarto de circulo, rejilla fina) da 0 pliegues; una fila con un doblez
insertado a proposito (avanza y se devuelve exactamente por el mismo camino) se detecta al 100%.

Uso:
  python wm/mesh_topology_check.py <mesh_dir_con_wNNN_o_x.tif_directo> [--min-cos -0.3] [--min-step 0.5]
Imprime una linea JSON por malla (wNNN) con: filas evaluadas, fraccion de vertices interiores
marcados como pliegue, la peor fila, y (si hay pliegues) hasta 5 ejemplos (fila, col, coseno).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def load_tifxyz(mesh_dir: str):
    """Lee x.tif/y.tif/z.tif de un directorio de malla tifxyz. Prueba tifffile primero (formatos
    de 32/64 bits que PIL a veces no abre bien, ver wm/scan_mesh_zones.py) y cae a PIL si falla."""
    paths = {c: os.path.join(mesh_dir, f"{c}.tif") for c in "xyz"}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    try:
        import tifffile
        arrs = {c: tifffile.imread(paths[c]).astype(np.float64) for c in "xyz"}
    except Exception:
        from PIL import Image
        arrs = {c: np.array(Image.open(paths[c]), np.float64) for c in "xyz"}
    x, y, z = arrs["x"], arrs["y"], arrs["z"]
    valid = (x > 0) & (y > 0) & (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x, y, z, valid


def row_fold_report(x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray,
                     min_step: float = 0.5, min_cos: float = -0.3) -> dict:
    """Pliegues por fila. Logica pura numpy, offline-testable (sin I/O).

    Para cada fila, toma las columnas validas EN ORDEN DE COLUMNA (no de "orden de aparicion" --
    una malla tifxyz ya esta indexada asi por construccion) y mide el coseno de giro en cada
    vertice interior con vecinos a distancia >= min_step en ambos lados. min_cos es el umbral por
    debajo del cual un vertice se marca como pliegue (mas negativo = giro mas cerrado).
    """
    n_rows, n_cols = x.shape
    row_stats = []
    examples = []
    total_interior, total_fold = 0, 0
    for r in range(n_rows):
        cols = np.flatnonzero(valid[r])
        if len(cols) < 3:
            continue
        pts = np.stack([x[r, cols], y[r, cols], z[r, cols]], axis=1)
        # pasos entre columnas validas CONSECUTIVAS en la rejilla (cols ya esta ordenado)
        step = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(step, axis=1)
        n_interior = len(cols) - 2
        if n_interior <= 0:
            continue
        e_in = step[:-1]
        e_out = step[1:]
        len_in, len_out = seg_len[:-1], seg_len[1:]
        measurable = (len_in >= min_step) & (len_out >= min_step)
        # cos(angle between the incoming and outgoing step direction): +1 = sigue derecho,
        # -1 = se devuelve exactamente por donde vino (pliegue perfecto)
        cosang = np.full(n_interior, np.nan)
        m = measurable
        if m.any():
            cosang[m] = (e_in[m] * e_out[m]).sum(1) / (len_in[m] * len_out[m])
        is_fold = m & (cosang < min_cos)
        n_meas = int(m.sum())
        n_fold = int(is_fold.sum())
        if n_meas:
            total_interior += n_meas
            total_fold += n_fold
        row_stats.append({"row": int(r), "n_cols_valid": int(len(cols)), "n_measurable": n_meas,
                           "n_fold": n_fold, "frac_fold": round(n_fold / n_meas, 4) if n_meas else None,
                           "min_cos": round(float(np.nanmin(cosang)), 4) if n_meas else None})
        for i in np.flatnonzero(is_fold):
            examples.append({"row": int(r), "col": int(cols[i + 1]), "cos": round(float(cosang[i]), 4)})
    worst = max((rs for rs in row_stats if rs["n_measurable"]), key=lambda rs: rs["frac_fold"], default=None)
    examples.sort(key=lambda e: e["cos"])
    return {
        "n_rows_with_data": len(row_stats),
        "n_rows_total": int(n_rows),
        "interior_vertices_measured": total_interior,
        "interior_vertices_folded": total_fold,
        "frac_folded": round(total_fold / total_interior, 5) if total_interior else None,
        "worst_row": worst,
        "fold_examples": examples[:5],
    }


def check_mesh_dir(mesh_dir: str, min_step: float = 0.5, min_cos: float = -0.3) -> dict:
    x, y, z, valid = load_tifxyz(mesh_dir)
    rep = row_fold_report(x, y, z, valid, min_step=min_step, min_cos=min_cos)
    rep["mesh_dir"] = mesh_dir
    rep["shape"] = list(x.shape)
    rep["n_valid_vertices"] = int(valid.sum())
    return rep


def summarize_windings(reports: list, min_vertices: int = 200, top_k: int = 5) -> dict:
    """Resume una lista de reportes por-vuelta (uno por wNNN, como los escribe check_mesh_dir /
    row_fold_report) en un solo veredicto por MALLA (todas sus vueltas juntas).

    Cierra el pendiente anotado en wm/MESH_TOPOLOGY_CHECK.md ("Proximo paso" #1): promediar la
    frac_folded de cada vuelta con el mismo peso deja que una vuelta angosta (pocas columnas, pocos
    vertices medidos -- ej. w000 de rodriguescarson con 41 vertices y 95% "pliegue") pese lo mismo
    que una vuelta grande con miles de vertices medidos, dando una fraccion global ruidosa y una
    lista de "peores vueltas" dominada por ruido en vez de senal real.

    Dos correcciones, ambas necesarias:
    1. La fraccion GLOBAL de la malla se calcula como sum(interior_vertices_folded) /
       sum(interior_vertices_measured) sobre TODAS las vueltas (ponderada por vertices de forma
       natural, no un promedio de fracciones) -- igual que ya se hacia a mano para la tabla de
       wm/MESH_TOPOLOGY_CHECK.md, aqui queda automatizado y reproducible.
    2. Al listar las vueltas con mas pliegues ("top offenders"), las vueltas con menos de
       `min_vertices` vertices interiores medidos se EXCLUYEN del ranking (van aparte, marcadas
       como dato insuficiente) en vez de competir por los primeros lugares con una fraccion que
       puede ser un artefacto de tener solo unas decenas de vertices medidos.
    """
    if not reports:
        return {"n_windings": 0, "total_interior_vertices_measured": 0,
                "total_interior_vertices_folded": 0, "frac_folded_weighted": None,
                "n_windings_with_fold": 0, "excluded_low_data": [], "top_windings": []}

    total_meas = sum(r.get("interior_vertices_measured", 0) or 0 for r in reports)
    total_fold = sum(r.get("interior_vertices_folded", 0) or 0 for r in reports)
    n_with_fold = sum(1 for r in reports if (r.get("interior_vertices_folded") or 0) > 0)

    ranked, excluded = [], []
    for r in reports:
        n_meas = r.get("interior_vertices_measured", 0) or 0
        entry = {"mesh_dir": r.get("mesh_dir"), "interior_vertices_measured": n_meas,
                 "frac_folded": r.get("frac_folded")}
        if n_meas < min_vertices:
            excluded.append(entry)
        else:
            ranked.append(entry)
    ranked.sort(key=lambda e: (e["frac_folded"] if e["frac_folded"] is not None else -1), reverse=True)
    excluded.sort(key=lambda e: e["mesh_dir"] or "")

    return {
        "n_windings": len(reports),
        "n_windings_low_data_excluded": len(excluded),
        "min_vertices_threshold": min_vertices,
        "total_interior_vertices_measured": total_meas,
        "total_interior_vertices_folded": total_fold,
        "frac_folded_weighted": round(total_fold / total_meas, 5) if total_meas else None,
        "n_windings_with_fold": n_with_fold,
        "top_windings": ranked[:top_k],
        "excluded_low_data": excluded,
    }


def _row_points_xy(x: np.ndarray, y: np.ndarray, valid: np.ndarray, row: int):
    """Puntos (col_valida, 2) en XY de una fila, EN ORDEN DE COLUMNA. None si hay <2 puntos
    (no se puede formar ni un segmento)."""
    cols = np.flatnonzero(valid[row])
    if len(cols) < 2:
        return None
    return np.stack([x[row, cols], y[row, cols]], axis=1)


def _row_mean_z(z: np.ndarray, valid: np.ndarray, row: int):
    cols = np.flatnonzero(valid[row])
    if len(cols) == 0:
        return None
    return float(np.mean(z[row, cols]))


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signo del producto cruzado (b-a) x (c-a) en 2D; positivo = c a la izquierda del rayo a->b."""
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _segment_crossing_matrix(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Matriz booleana (nA-1, nB-1): [i, j] es True si el segmento i de la polilinea A CRUZA
    (en sentido estricto -- se excluyen a proposito los casos de contacto en un extremo o
    colineales, para no marcar falsos positivos por coincidencias de punto flotante) el
    segmento j de la polilinea B. Test de orientacion clasico (signos opuestos de ambos lados),
    vectorizado con numpy: nA y nB son chicos en mallas tifxyz reales (decenas de columnas por
    fila, ver wm/data_topology/*.jsonl), asi que la matriz completa es barata."""
    if len(pts_a) < 2 or len(pts_b) < 2:
        return np.zeros((max(len(pts_a) - 1, 0), max(len(pts_b) - 1, 0)), dtype=bool)
    a1, a2 = pts_a[:-1, None, :], pts_a[1:, None, :]   # (nA-1, 1, 2)
    b1, b2 = pts_b[None, :-1, :], pts_b[None, 1:, :]   # (1, nB-1, 2)
    d1 = _orient(b1, b2, a1)
    d2 = _orient(b1, b2, a2)
    d3 = _orient(a1, a2, b1)
    d4 = _orient(a1, a2, b2)
    return (((d1 > 0) != (d2 > 0)) & (d1 != 0) & (d2 != 0) &
            ((d3 > 0) != (d4 > 0)) & (d3 != 0) & (d4 != 0))


def neighbor_crossing_report(x_a: np.ndarray, y_a: np.ndarray, z_a: np.ndarray, valid_a: np.ndarray,
                              x_b: np.ndarray, y_b: np.ndarray, z_b: np.ndarray, valid_b: np.ndarray,
                              z_tol: float = 2.0) -> dict:
    """Frente G, candidata 2, paso 2 del plan (wm/MESH_TOPOLOGY_CHECK.md): intento de responder
    "¿se cruzan dos VUELTAS VECINAS (wNNN y wNNN+1) en 3D?".

    *** INVALIDADO por un chequeo de control (27 sep 2026, turno automático) -- NO USAR sus
    numeros como si midieran cruces reales. Ver wm/MESH_TOPOLOGY_CHECK.md seccion "Paso 2
    (INVALIDADO)" para el detalle completo. Se deja el codigo (probado y correcto para LO QUE
    HACE: detectar cruces exactos entre dos polilineas 2D ya dadas -- ver los 12 tests sinteticos
    en wm/tests/test_mesh_topology_check.py, todos verdes) como base para un rediseno futuro, NO
    como una herramienta lista para usar. ***

    Por que esta invalidado: el metodo compara, para cada fila de A, la fila de B cuya Z REAL
    promedio (de las columnas validas) sea la mas cercana dentro de z_tol voxeles, proyectando
    ambas "rebanadas" a 2D (XY) e ignorando la Z residual de cada punto. Esto asume que una fila
    es aproximadamente un corte plano en Z -- FALSO en datos reales: en el par de control
    PHerc0139 w034/w035 (curado, con 0.0% de pliegues internos segun row_fold_report), la Z
    varia HASTA 150-235 voxeles ENTRE COLUMNAS DE UNA MISMA FILA (mucho mas que z_tol=2.0), y la
    distancia 3D real minima entre los puntos de las dos filas "emparejadas por Z" resulto ser de
    12 a 700+ voxeles (nada cercano a un cruce real) -- aun asi, el test de cruce 2D proyectado
    marco 94%/93% de las filas emparejadas como "con cruce" en AMBOS pares de control curados
    (w034-w035 y w035-w036). Un chequeo que marca la inmensa mayoria de una malla SANA conocida
    como "cruzada" no mide lo que dice medir -- es un artefacto de proyectar a 2D descartando una
    Z que en realidad varia mucho, no evidencia de que las vueltas realmente se toquen o crucen.

    Rediseno pendiente (ver "Proximo paso" en wm/MESH_TOPOLOGY_CHECK.md): comparar vertices en 3D
    de verdad (p. ej. distancia minima de cada vertice de A al vertice/arista mas cercano de TODA
    la malla B via KD-tree, no solo la fila con Z promedio mas cercana) y buscar caidas ANOMALAS
    de esa distancia local respecto al hueco tipico entre las dos vueltas -- no un cruce en un
    plano proyectado que nunca existio como tal en los datos.

    Logica pura numpy, offline-testable (sin I/O), igual que row_fold_report (eso SI sigue siendo
    valido -- lo que fallo es la idea de comparar por fila-Z-mas-cercana-proyectada-a-2D, no la
    funcion de interseccion de segmentos en si).
    """
    rows_a = [r for r in range(x_a.shape[0]) if np.count_nonzero(valid_a[r]) >= 2]
    rows_b = [r for r in range(x_b.shape[0]) if np.count_nonzero(valid_b[r]) >= 2]
    z_means_b = {r: _row_mean_z(z_b, valid_b, r) for r in rows_b}
    z_means_b = {r: z for r, z in z_means_b.items() if z is not None}

    matched, examples, total_crossings = [], [], 0
    for ra in rows_a:
        za = _row_mean_z(z_a, valid_a, ra)
        if za is None or not z_means_b:
            continue
        best_rb = min(z_means_b, key=lambda r: abs(z_means_b[r] - za))
        best_gap = abs(z_means_b[best_rb] - za)
        if best_gap > z_tol:
            continue
        pts_a = _row_points_xy(x_a, y_a, valid_a, ra)
        pts_b = _row_points_xy(x_b, y_b, valid_b, best_rb)
        if pts_a is None or pts_b is None:
            continue
        # rechazo rapido: cajas (bounding boxes) en XY que no se tocan no pueden cruzarse
        if (pts_a[:, 0].max() < pts_b[:, 0].min() or pts_b[:, 0].max() < pts_a[:, 0].min() or
                pts_a[:, 1].max() < pts_b[:, 1].min() or pts_b[:, 1].max() < pts_a[:, 1].min()):
            n_cross, cmat = 0, None
        else:
            cmat = _segment_crossing_matrix(pts_a, pts_b)
            n_cross = int(cmat.sum())
        matched.append({"row_a": int(ra), "row_b": int(best_rb), "z_gap": round(best_gap, 3),
                         "n_crossings": n_cross})
        if n_cross:
            total_crossings += n_cross
            ia, ib = np.argwhere(cmat)[0]
            examples.append({"row_a": int(ra), "row_b": int(best_rb), "z_gap": round(best_gap, 3),
                              "n_crossings_this_row": n_cross,
                              "point_a": [round(float(v), 2) for v in pts_a[ia]],
                              "point_b": [round(float(v), 2) for v in pts_b[ib]]})

    n_matched = len(matched)
    n_crossing_rows = sum(1 for m in matched if m["n_crossings"] > 0)
    examples.sort(key=lambda e: -e["n_crossings_this_row"])
    return {
        "n_rows_matched": n_matched,
        "n_rows_with_crossing": n_crossing_rows,
        "frac_rows_with_crossing": round(n_crossing_rows / n_matched, 4) if n_matched else None,
        "total_crossings": total_crossings,
        "z_tol": z_tol,
        "crossing_examples": examples[:5],
    }


def check_neighbor_pair(mesh_dir_a: str, mesh_dir_b: str, z_tol: float = 2.0) -> dict:
    x_a, y_a, z_a, valid_a = load_tifxyz(mesh_dir_a)
    x_b, y_b, z_b, valid_b = load_tifxyz(mesh_dir_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=z_tol)
    rep["mesh_dir_a"] = mesh_dir_a
    rep["mesh_dir_b"] = mesh_dir_b
    return rep


def neighbor_gap_report(x_a: np.ndarray, y_a: np.ndarray, z_a: np.ndarray, valid_a: np.ndarray,
                         x_b: np.ndarray, y_b: np.ndarray, z_b: np.ndarray, valid_b: np.ndarray,
                         window_rows: int = 5, anomaly_frac: float = 0.25,
                         min_neighbor_rows: int = 2) -> dict:
    """Frente G, candidata 2, REDISENO del paso 2 (27 sep 2026, turno automatico) tras invalidar
    `neighbor_crossing_report` (ver su docstring y wm/MESH_TOPOLOGY_CHECK.md seccion "Paso 2
    INVALIDADO"): en vez de emparejar filas por Z-promedio-mas-cercana y proyectar a 2D (premisa
    falsa: la Z real varia 150-235 vox DENTRO de una misma fila), esta version mide la distancia
    3D REAL de cada vertice de A al vertice MAS CERCANO de TODA la malla B (KD-tree sobre la nube
    de puntos de B, sin proyectar nada) y busca caidas ANOMALAS de esa distancia respecto al
    "hueco tipico" LOCAL entre las dos vueltas en esa zona -- exactamente el rediseno que quedo
    pendiente documentado.

    "Local" importa: el hueco entre dos vueltas vecinas varia de forma suave a lo largo del
    rollo (las vueltas internas estan mas comprimidas que las externas), asi que comparar contra
    un hueco GLOBAL marcaria como "anomalo" cualquier zona donde el hueco tipico ya es chico de
    por si. En cambio, la linea base de una fila `r` es la MEDIANA del hueco tipico (mediana por
    fila de las distancias vertice-a-vertice-mas-cercano) de las filas VECINAS (indice de fila a
    <= window_rows de distancia, EXCLUYENDO la propia fila `r` para que una fila ya anomala no
    contamine su propia linea base). Un vertice se marca anomalo si su hueco es menor a
    `anomaly_frac` de esa linea base local -- una caida real (p. ej. 0.25 = el hueco se redujo a
    menos de 1/4 de lo esperado ahi mismo), no solo "mas chico que el promedio de todo el rollo".

    Filas sin suficientes filas vecinas con datos (< min_neighbor_rows, tipico en los bordes de
    la malla) se dejan SIN linea base (no se marcan ni a favor ni en contra) en vez de arriesgar
    un falso positivo/negativo por una linea base mal estimada.

    Logica pura numpy + scipy.spatial.cKDTree (offline, sin descargar nada nuevo una vez que se
    tienen las dos mallas), igual de testable con datos sinteticos que row_fold_report.
    """
    from scipy.spatial import cKDTree

    rows_b_has_data = np.count_nonzero(valid_b) if valid_b is not None else 0
    if not rows_b_has_data:
        return {"n_rows_a": 0, "n_rows_with_baseline": 0, "vertices_measured": 0,
                "vertices_anomalous": 0, "frac_anomalous": None, "anomaly_frac_threshold": anomaly_frac,
                "window_rows": window_rows, "worst_row": None, "anomaly_examples": [],
                "error": "no valid data in mesh B"}

    pts_b = np.stack([x_b[valid_b], y_b[valid_b], z_b[valid_b]], axis=1)
    tree = cKDTree(pts_b)

    rows_a = [r for r in range(x_a.shape[0]) if np.count_nonzero(valid_a[r]) >= 1]
    row_gaps, row_median = {}, {}
    for r in rows_a:
        cols = np.flatnonzero(valid_a[r])
        pts_a_row = np.stack([x_a[r, cols], y_a[r, cols], z_a[r, cols]], axis=1)
        dist, _ = tree.query(pts_a_row)
        row_gaps[r] = dist
        row_median[r] = float(np.median(dist))

    sorted_rows = sorted(row_median)
    total_meas, total_anom = 0, 0
    anomaly_examples, row_report = [], []
    for r in sorted_rows:
        neighbor_meds = [row_median[rr] for rr in sorted_rows
                         if rr != r and abs(rr - r) <= window_rows]
        gaps = row_gaps[r]
        cols = np.flatnonzero(valid_a[r])
        if len(neighbor_meds) < min_neighbor_rows:
            row_report.append({"row": int(r), "n_measured": int(len(gaps)),
                               "baseline_gap": None, "n_anomaly": None})
            continue
        baseline = float(np.median(neighbor_meds))
        total_meas += len(gaps)
        if baseline <= 0:
            row_report.append({"row": int(r), "n_measured": int(len(gaps)),
                               "baseline_gap": baseline, "n_anomaly": 0})
            continue
        is_anom = gaps < anomaly_frac * baseline
        n_anom = int(is_anom.sum())
        total_anom += n_anom
        row_report.append({"row": int(r), "n_measured": int(len(gaps)),
                           "baseline_gap": round(baseline, 3), "n_anomaly": n_anom,
                           "min_gap": round(float(gaps.min()), 3)})
        for j in np.flatnonzero(is_anom):
            anomaly_examples.append({"row": int(r), "col": int(cols[j]),
                                     "gap": round(float(gaps[j]), 3),
                                     "baseline_gap": round(baseline, 3)})

    with_baseline = [rr for rr in row_report if rr["baseline_gap"] is not None]
    worst_row = max(with_baseline,
                    key=lambda rr: (rr["n_anomaly"] / rr["n_measured"]) if rr["n_measured"] else 0,
                    default=None)
    anomaly_examples.sort(key=lambda e: e["gap"])
    return {
        "n_rows_a": len(sorted_rows),
        "n_rows_with_baseline": len(with_baseline),
        "vertices_measured": total_meas,
        "vertices_anomalous": total_anom,
        "frac_anomalous": round(total_anom / total_meas, 5) if total_meas else None,
        "anomaly_frac_threshold": anomaly_frac,
        "window_rows": window_rows,
        "worst_row": worst_row,
        "anomaly_examples": anomaly_examples[:5],
    }


def check_neighbor_gap_pair(mesh_dir_a: str, mesh_dir_b: str, window_rows: int = 5,
                             anomaly_frac: float = 0.25) -> dict:
    x_a, y_a, z_a, valid_a = load_tifxyz(mesh_dir_a)
    x_b, y_b, z_b, valid_b = load_tifxyz(mesh_dir_b)
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b,
                               window_rows=window_rows, anomaly_frac=anomaly_frac)
    rep["mesh_dir_a"] = mesh_dir_a
    rep["mesh_dir_b"] = mesh_dir_b
    return rep


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="directorio con x.tif/y.tif/z.tif, un directorio padre con "
                                 "subcarpetas wNNN (cada una revisada por separado), o -- con "
                                 "--summarize -- un archivo .jsonl con un reporte por vuelta ya "
                                 "generado (una linea de check_mesh_dir por wNNN)")
    ap.add_argument("path_b", nargs="?", default=None,
                    help="si se da, corre un chequeo entre dos vueltas vecinas path=wNNN y "
                         "path_b=wNNN+1 (cual, ver --method)")
    ap.add_argument("--method", choices=["gap", "crossing2d"], default="gap",
                    help="con path_b: 'gap' (default, VALIDADO 27 sep 2026 contra control real) = "
                         "distancia 3D real via KD-tree con linea base local (neighbor_gap_report); "
                         "'crossing2d' = EXPERIMENTAL, INVALIDADO en datos de control, ver docstring "
                         "de neighbor_crossing_report -- solo para referencia, no usar sus numeros")
    ap.add_argument("--min-step", type=float, default=0.5, help="voxeles minimos de paso para medir un angulo")
    ap.add_argument("--min-cos", type=float, default=-0.3, help="coseno bajo el cual un vertice cuenta como pliegue")
    ap.add_argument("--z-tol", type=float, default=2.0,
                    help="con path_b --method crossing2d: voxeles maximos de diferencia en Z real "
                         "para emparejar una fila de A con una de B")
    ap.add_argument("--window-rows", type=int, default=5,
                    help="con path_b --method gap: filas vecinas (por indice) usadas para la linea base local")
    ap.add_argument("--anomaly-frac", type=float, default=0.25,
                    help="con path_b --method gap: un vertice es anomalo si su hueco es menor a esta "
                         "fraccion de la linea base local")
    ap.add_argument("--summarize", action="store_true",
                    help="en vez de correr el chequeo, resume un .jsonl de reportes por-vuelta ya "
                         "existente en un veredicto ponderado por malla (ver summarize_windings)")
    ap.add_argument("--min-vertices", type=int, default=200,
                    help="con --summarize: vueltas con menos vertices interiores medidos que esto "
                         "se excluyen del ranking de 'peores vueltas' (dato insuficiente)")
    a = ap.parse_args()

    if a.summarize:
        with open(a.path) as f:
            reports = [json.loads(line) for line in f if line.strip()]
        print(json.dumps(summarize_windings(reports, min_vertices=a.min_vertices), indent=2))
        return

    if a.path_b:
        if a.method == "gap":
            rep = check_neighbor_gap_pair(a.path, a.path_b, window_rows=a.window_rows,
                                          anomaly_frac=a.anomaly_frac)
        else:
            rep = check_neighbor_pair(a.path, a.path_b, z_tol=a.z_tol)
        print(json.dumps(rep))
        return

    sub = sorted(d for d in glob.glob(os.path.join(a.path, "w[0-9][0-9][0-9]")) if os.path.isdir(d))
    targets = sub if sub else [a.path]
    for t in targets:
        try:
            rep = check_mesh_dir(t, min_step=a.min_step, min_cos=a.min_cos)
        except FileNotFoundError as e:
            print(json.dumps({"mesh_dir": t, "error": f"missing {e}"}))
            continue
        print(json.dumps(rep))


if __name__ == "__main__":
    _cli()
