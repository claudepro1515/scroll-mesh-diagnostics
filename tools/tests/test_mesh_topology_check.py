import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mesh_topology_check import (  # noqa: E402
    row_fold_report, load_tifxyz, summarize_windings,
    neighbor_crossing_report, check_neighbor_pair, _segment_crossing_matrix,
    neighbor_gap_report, check_neighbor_gap_pair,
)


def _mesh_from_row(row_xyz, n_rows=3, row_spacing=5.0):
    """Repite una sola fila (n_cols, 3) en n_rows filas identicas salvo un offset en Z, todas validas."""
    n_cols = row_xyz.shape[0]
    x = np.tile(row_xyz[:, 0], (n_rows, 1))
    y = np.tile(row_xyz[:, 1], (n_rows, 1))
    z = np.tile(row_xyz[:, 2], (n_rows, 1)) + row_spacing * np.arange(n_rows)[:, None]
    valid = np.ones((n_rows, n_cols), bool)
    return x, y, z, valid


def test_smooth_arc_has_no_folds():
    # cuarto de circulo, radio 500, 200 columnas -- paso angular fino, sin pliegues reales
    theta = np.linspace(0, np.pi / 2, 200)
    row = np.stack([500 * np.cos(theta), 500 * np.sin(theta), np.zeros_like(theta)], axis=1)
    x, y, z, valid = _mesh_from_row(row)
    rep = row_fold_report(x, y, z, valid)
    assert rep["frac_folded"] == 0.0
    assert rep["fold_examples"] == []


def test_straight_line_has_no_folds():
    row = np.stack([np.arange(100, dtype=float), np.zeros(100), np.zeros(100)], axis=1)
    x, y, z, valid = _mesh_from_row(row)
    rep = row_fold_report(x, y, z, valid)
    assert rep["frac_folded"] == 0.0


def test_inserted_fold_is_detected():
    # avanza en linea recta 0..49, luego se devuelve EXACTAMENTE por el mismo camino 48..0:
    # un pliegue perfecto (coseno = -1) justo en el vertice de retorno (col 50, x=49)
    out = np.arange(0, 50, dtype=float)
    back = np.arange(48, -1, -1, dtype=float)
    xs = np.concatenate([out, back])
    row = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    x, y, z, valid = _mesh_from_row(row)
    rep = row_fold_report(x, y, z, valid)
    assert rep["frac_folded"] > 0
    assert rep["interior_vertices_folded"] == 3  # una por fila (3 filas identicas)
    # el pliegue esta en la columna de vuelta (indice 50, x=49 duplicado con el indice 49)
    assert all(e["cos"] < -0.99 for e in rep["fold_examples"])


def test_gentle_curve_below_threshold_not_flagged():
    # un giro de 60 grados (coseno de la vuelta = -0.5, o sea angulo entre vectores de 120,
    # bastante mas cerrado que una curva suave real) SI se marca con el umbral por defecto
    # (-0.3); aqui probamos un giro suave de 20 grados entre pasos, que NO debe marcarse
    theta_step = np.deg2rad(20)
    pts = [np.array([0.0, 0.0, 0.0])]
    d = np.array([1.0, 0.0, 0.0])
    for _ in range(20):
        pts.append(pts[-1] + d * 5.0)
        c, s = np.cos(theta_step), np.sin(theta_step)
        d = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, 0.0])
    row = np.stack(pts)
    x, y, z, valid = _mesh_from_row(row)
    rep = row_fold_report(x, y, z, valid)
    assert rep["frac_folded"] == 0.0


def test_short_steps_excluded_from_denominator():
    # dos puntos casi identicos (paso < min_step) no deben contarse como medibles ni como pliegue
    row = np.array([[0.0, 0, 0], [0.1, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    x, y, z, valid = _mesh_from_row(row, n_rows=1)
    rep = row_fold_report(x, y, z, valid, min_step=0.5)
    # el primer vertice interior (col 1, entre pasos 0.1 y 9.9) tiene paso de entrada < min_step
    assert rep["interior_vertices_measured"] == 1  # solo el vertice col=2 es medible
    assert rep["frac_folded"] == 0.0


def test_invalid_columns_are_skipped_not_treated_as_adjacent():
    # fila con un hueco invalido en medio; las columnas validas deben tratarse como vecinas
    # entre si (columna 10 valida sigue a la columna 4 valida), no como un salto grande erroneo
    xs = np.arange(20, dtype=float)
    row = np.stack([xs, np.zeros(20), np.zeros(20)], axis=1)
    x, y, z, valid = _mesh_from_row(row, n_rows=1)
    valid[0, 5:10] = False  # invalida columnas 5..9
    rep = row_fold_report(x, y, z, valid)
    # sigue siendo una linea recta en las columnas validas restantes -> sin pliegues
    assert rep["frac_folded"] == 0.0


def test_all_invalid_row_skipped_gracefully():
    x = np.zeros((2, 10))
    y = np.zeros((2, 10))
    z = np.zeros((2, 10))
    valid = np.zeros((2, 10), bool)
    rep = row_fold_report(x, y, z, valid)
    assert rep["interior_vertices_measured"] == 0
    assert rep["frac_folded"] is None
    assert rep["worst_row"] is None


def test_load_tifxyz_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tifxyz(str(tmp_path))


def _fake_report(mesh_dir, n_measured, n_folded):
    frac = round(n_folded / n_measured, 5) if n_measured else None
    return {"mesh_dir": mesh_dir, "interior_vertices_measured": n_measured,
            "interior_vertices_folded": n_folded, "frac_folded": frac}


def test_summarize_windings_weights_by_vertex_count_not_by_winding():
    # w000 es angosta (41 medidos) con una fraccion ENORME (95%) -- puro ruido de muestra chica,
    # igual que el caso real documentado (rodriguescarson PHerc0826 w000). Las demas vueltas son
    # grandes (miles de vertices) con una fraccion real baja. Promediar las fracciones de cada
    # vuelta con el mismo peso daria un resultado dominado por el ruido de w000; ponderar por
    # vertices medidos (sum folded / sum measured) no.
    reports = [
        _fake_report("w000", 41, 39),      # 95.1%, ruido de muestra chica
        _fake_report("w001", 3000, 30),    # 1.0%
        _fake_report("w002", 3000, 30),    # 1.0%
        _fake_report("w003", 3000, 30),    # 1.0%
    ]
    naive_mean_of_fractions = sum(r["frac_folded"] for r in reports) / len(reports)
    summary = summarize_windings(reports, min_vertices=200)
    # la fraccion global ponderada por vertices debe ser MUCHO mas baja que el promedio ingenuo
    # de las 4 fracciones (que arrastra el 95% de la vuelta angosta)
    assert summary["frac_folded_weighted"] < naive_mean_of_fractions / 10
    assert summary["frac_folded_weighted"] == round((39 + 30 + 30 + 30) / (41 + 3000 + 3000 + 3000), 5)


def test_summarize_windings_excludes_low_data_from_ranking():
    reports = [
        _fake_report("w000", 41, 39),       # 95.1%, pero por debajo de min_vertices -> excluida
        _fake_report("w010", 2281, 384),    # 16.8%, caso real (armando w010)
        _fake_report("w020", 10000, 5),     # 0.05%, vuelta sana
    ]
    summary = summarize_windings(reports, min_vertices=200, top_k=5)
    excluded_ids = [e["mesh_dir"] for e in summary["excluded_low_data"]]
    top_ids = [e["mesh_dir"] for e in summary["top_windings"]]
    assert excluded_ids == ["w000"]
    assert top_ids[0] == "w010"  # la peor vuelta CON datos suficientes queda primera
    assert "w000" not in top_ids
    assert summary["n_windings_low_data_excluded"] == 1


def test_summarize_windings_empty_list():
    summary = summarize_windings([])
    assert summary["n_windings"] == 0
    assert summary["frac_folded_weighted"] is None
    assert summary["top_windings"] == []
    assert summary["excluded_low_data"] == []


def test_summarize_windings_top_k_respected():
    reports = [_fake_report(f"w{i:03d}", 1000, i * 10) for i in range(10)]
    summary = summarize_windings(reports, min_vertices=200, top_k=3)
    assert len(summary["top_windings"]) == 3
    # orden descendente por frac_folded: la de mas pliegue (i=9) primero
    assert summary["top_windings"][0]["mesh_dir"] == "w009"


# ---------------------------------------------------------------------------
# Frente G, candidata 2, paso 2 (wm/MESH_TOPOLOGY_CHECK.md "Proximo paso" #2):
# cruce entre dos vueltas VECINAS (wNNN vs wNNN+1). Pruebas sinteticas ANTES de
# tocar datos reales -- mismo orden que row_fold_report en su momento.
#
# NOTA (27 sep 2026, turno automático, DESPUES de estas pruebas): el diseno que estas pruebas
# verifican (emparejar filas por Z promedio mas cercana y comparar como polilineas 2D) quedo
# INVALIDADO por un chequeo de control con datos reales (94%/93% de filas "con cruce" en pares
# de control curados sanos, PHerc0139 w034-w035/w035-w036 -- ver docstring de
# neighbor_crossing_report y wm/MESH_TOPOLOGY_CHECK.md). Estas pruebas siguen siendo correctas
# para lo que verifican (la mecanica de _segment_crossing_matrix y el emparejamiento por Z son
# EXACTAMENTE lo que dicen ser) -- lo que fallo no es un bug de codigo sino que esa mecanica no
# es una buena proxy de "cruce real en 3D" con datos reales, donde la Z varia mucho dentro de una
# misma fila. Se dejan como caracterizacion de la funcion pura mientras se redisena el metodo.
# ---------------------------------------------------------------------------

def _grid_from_rows(rows_xyz):
    """Arma una rejilla (x, y, z, valid) a partir de una lista de filas (n_cols_i, 3), cada una
    con su propio Z real (a diferencia de _mesh_from_row, que repite una sola fila con un offset
    fijo) -- necesario para construir pares de mallas con Z's que no coinciden exactamente entre
    si, o en orden de fila distinto, y probar que el emparejamiento es por Z real, no por indice."""
    n_rows = len(rows_xyz)
    n_cols = max(r.shape[0] for r in rows_xyz)
    x = np.zeros((n_rows, n_cols))
    y = np.zeros((n_rows, n_cols))
    z = np.zeros((n_rows, n_cols))
    valid = np.zeros((n_rows, n_cols), bool)
    for i, row in enumerate(rows_xyz):
        n = row.shape[0]
        x[i, :n], y[i, :n], z[i, :n] = row[:, 0], row[:, 1], row[:, 2]
        valid[i, :n] = True
    return x, y, z, valid


def _line_row(xs, ys, z_val):
    return np.stack([xs, ys, np.full_like(xs, float(z_val))], axis=1)


def test_segment_crossing_matrix_detects_simple_x_crossing():
    pts_a = np.array([[0.0, 0.0], [2.0, 2.0]])
    pts_b = np.array([[0.0, 2.0], [2.0, 0.0]])
    cmat = _segment_crossing_matrix(pts_a, pts_b)
    assert cmat.shape == (1, 1)
    assert bool(cmat[0, 0]) is True


def test_segment_crossing_matrix_parallel_segments_no_crossing():
    pts_a = np.array([[0.0, 0.0], [10.0, 0.0]])
    pts_b = np.array([[0.0, 5.0], [10.0, 5.0]])
    assert not _segment_crossing_matrix(pts_a, pts_b).any()


def test_segment_crossing_matrix_shared_endpoint_not_flagged():
    # se tocan justo en (10, 0) -- un contacto en un extremo compartido, no un cruce transversal
    # real; se excluye a proposito (ver docstring) para no marcar falsos positivos de punto flotante
    pts_a = np.array([[0.0, 0.0], [10.0, 0.0]])
    pts_b = np.array([[10.0, 0.0], [10.0, 10.0]])
    assert not _segment_crossing_matrix(pts_a, pts_b).any()


def test_neighbor_crossing_report_parallel_lines_no_crossing():
    xs = np.arange(10, dtype=float)
    rows_a = [_line_row(xs, np.zeros(10), z) for z in (0.0, 10.0, 20.0)]
    rows_b = [_line_row(xs, np.full(10, 5.0), z) for z in (0.0, 10.0, 20.0)]
    x_a, y_a, z_a, valid_a = _grid_from_rows(rows_a)
    x_b, y_b, z_b, valid_b = _grid_from_rows(rows_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=1.0)
    assert rep["n_rows_matched"] == 3
    assert rep["total_crossings"] == 0
    assert rep["frac_rows_with_crossing"] == 0.0
    assert rep["crossing_examples"] == []


def test_neighbor_crossing_report_detects_real_crossing_in_one_row():
    # fila z=0: B paralela a A (offset en y, sin cruce). Fila z=10: B pasa de un lado al otro de
    # A (de y=-5 a y=5 mientras A se queda en y=0) -- un cruce real de dos hojas vecinas
    xs = np.arange(10, dtype=float)
    rows_a = [_line_row(xs, np.zeros(10), 0.0), _line_row(xs, np.zeros(10), 10.0)]
    rows_b = [_line_row(xs, np.full(10, 5.0), 0.0), _line_row(xs, np.linspace(-5, 5, 10), 10.0)]
    x_a, y_a, z_a, valid_a = _grid_from_rows(rows_a)
    x_b, y_b, z_b, valid_b = _grid_from_rows(rows_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=1.0)
    assert rep["n_rows_matched"] == 2
    assert rep["n_rows_with_crossing"] == 1
    assert rep["total_crossings"] >= 1
    assert rep["crossing_examples"][0]["row_a"] == 1
    assert rep["crossing_examples"][0]["row_b"] == 1


def test_neighbor_crossing_report_matches_by_real_z_value_not_row_index():
    # A guarda sus filas en orden z=0,10,20 (indices 0,1,2). B guarda las MISMAS z's pero en
    # orden inverso (indices 0,1,2 -> z=20,10,0) -- si el emparejamiento fuera por indice de
    # fila (en vez de por Z real) el cruce (solo en z=20) se buscaria en el par de filas
    # equivocado. El cruce solo debe aparecer emparejado con z_gap == 0 en los indices reales
    # donde esa Z vive en cada malla (row_a=2 en A, row_b=0 en B).
    xs = np.arange(10, dtype=float)
    rows_a = [_line_row(xs, np.zeros(10), z) for z in (0.0, 10.0, 20.0)]
    row_b_z20_crossing = _line_row(xs, np.linspace(-5, 5, 10), 20.0)
    row_b_z10 = _line_row(xs, np.full(10, 5.0), 10.0)
    row_b_z0 = _line_row(xs, np.full(10, 5.0), 0.0)
    rows_b = [row_b_z20_crossing, row_b_z10, row_b_z0]  # orden invertido a proposito
    x_a, y_a, z_a, valid_a = _grid_from_rows(rows_a)
    x_b, y_b, z_b, valid_b = _grid_from_rows(rows_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=0.5)
    assert rep["n_rows_matched"] == 3
    assert rep["n_rows_with_crossing"] == 1
    ex = rep["crossing_examples"][0]
    assert ex["row_a"] == 2   # z=20 vive en el indice 2 de A
    assert ex["row_b"] == 0   # z=20 vive en el indice 0 de B (guardada primero)
    assert ex["z_gap"] == 0


def test_neighbor_crossing_report_respects_z_tol():
    xs = np.arange(10, dtype=float)
    rows_a = [_line_row(xs, np.zeros(10), 0.0)]
    rows_b = [_line_row(xs, np.full(10, 5.0), 100.0)]  # muy lejos en Z real
    x_a, y_a, z_a, valid_a = _grid_from_rows(rows_a)
    x_b, y_b, z_b, valid_b = _grid_from_rows(rows_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=2.0)
    assert rep["n_rows_matched"] == 0
    assert rep["frac_rows_with_crossing"] is None
    assert rep["total_crossings"] == 0


def test_neighbor_crossing_report_concentric_arcs_no_crossing():
    # dos arcos concentricos (radios distintos, mismo centro) en la MISMA Z -- capas anidadas
    # que nunca se tocan, como deberian verse dos vueltas vecinas sanas de un espiral
    theta = np.linspace(0, np.pi / 2, 60)
    row_a = np.stack([100 * np.cos(theta), 100 * np.sin(theta), np.zeros_like(theta)], axis=1)
    row_b = np.stack([110 * np.cos(theta), 110 * np.sin(theta), np.zeros_like(theta)], axis=1)
    x_a, y_a, z_a, valid_a = _grid_from_rows([row_a])
    x_b, y_b, z_b, valid_b = _grid_from_rows([row_b])
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=1.0)
    assert rep["n_rows_matched"] == 1
    assert rep["total_crossings"] == 0


def test_neighbor_crossing_report_bbox_fast_reject_no_crash():
    xs = np.arange(10, dtype=float)
    rows_a = [_line_row(xs, np.zeros(10), 0.0)]
    rows_b = [_line_row(xs + 1000.0, np.zeros(10), 0.0)]  # muy lejos en XY, cajas no se tocan
    x_a, y_a, z_a, valid_a = _grid_from_rows(rows_a)
    x_b, y_b, z_b, valid_b = _grid_from_rows(rows_b)
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=1.0)
    assert rep["n_rows_matched"] == 1
    assert rep["total_crossings"] == 0


def test_neighbor_crossing_report_skips_rows_with_too_few_points():
    # una fila con un solo punto valido no puede formar un segmento -- se salta, no revienta
    x_a = np.array([[0.0, 1.0]])
    y_a = np.array([[0.0, 0.0]])
    z_a = np.array([[0.0, 0.0]])
    valid_a = np.array([[True, False]])
    xs = np.arange(10, dtype=float)
    x_b, y_b, z_b, valid_b = _grid_from_rows([_line_row(xs, np.full(10, 5.0), 0.0)])
    rep = neighbor_crossing_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, z_tol=1.0)
    assert rep["n_rows_matched"] == 0


def test_check_neighbor_pair_reads_two_dirs(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile

    xs = np.arange(10, dtype=float)
    dir_a, dir_b = tmp_path / "w010", tmp_path / "w011"
    dir_a.mkdir()
    dir_b.mkdir()
    # load_tifxyz trata z<=0 (y x<=0, y<=0) como "sin dato" (mascara de mallas tifxyz reales,
    # ver load_tifxyz) -- z_val debe ser > 0 para que la fila cuente como valida al releerla
    x_a, y_a, z_a, _ = _grid_from_rows([_line_row(xs + 1, np.full(10, 1.0), 1.0)])
    x_b, y_b, z_b, _ = _grid_from_rows([_line_row(xs + 1, np.full(10, 5.0), 1.0)])
    for name, arr in (("x", x_a), ("y", y_a), ("z", z_a)):
        tifffile.imwrite(str(dir_a / f"{name}.tif"), arr.astype(np.float32))
    for name, arr in (("x", x_b), ("y", y_b), ("z", z_b)):
        tifffile.imwrite(str(dir_b / f"{name}.tif"), arr.astype(np.float32))
    rep = check_neighbor_pair(str(dir_a), str(dir_b), z_tol=1.0)
    assert rep["mesh_dir_a"] == str(dir_a)
    assert rep["mesh_dir_b"] == str(dir_b)
    assert rep["n_rows_matched"] == 1
    assert rep["total_crossings"] == 0


# ---------------------------------------------------------------------------------------------
# neighbor_gap_report: REDISENO del paso 2 tras invalidar neighbor_crossing_report (ver su
# docstring). Distancia 3D real via KD-tree contra TODA la malla B + linea base LOCAL (filas
# vecinas), no un cruce 2D proyectado.
# ---------------------------------------------------------------------------------------------

def _plane_mesh(rows=20, cols=30, x_scale=3.0, y_scale=5.0, z_offset=0.0):
    """Rejilla plana (rows x cols) en XY con Z constante (+ z_offset), toda valida."""
    row_idx = np.arange(rows)[:, None] * np.ones((1, cols))
    col_idx = np.ones((rows, 1)) * np.arange(cols)[None, :]
    x = col_idx * x_scale
    y = row_idx * y_scale
    z = np.full((rows, cols), float(z_offset))
    valid = np.ones((rows, cols), bool)
    return x, y, z, valid


def test_neighbor_gap_report_constant_gap_no_anomaly():
    # dos planos paralelos, separados 10 en Z de punta a punta -- ningun vertice deberia marcarse
    x_a, y_a, z_a, valid_a = _plane_mesh(z_offset=0.0)
    x_b, y_b, z_b, valid_b = _plane_mesh(z_offset=10.0)
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b, window_rows=3)
    assert rep["frac_anomalous"] == 0.0
    assert rep["anomaly_examples"] == []


def test_neighbor_gap_report_detects_local_dip():
    # plano B paralelo a A a distancia 10, salvo la fila 10 donde B casi TOCA a A (hueco 0.3) --
    # una caida local real, rodeada de hueco normal (esto es lo que el chequeo invalidado NO
    # podia distinguir de una fila cualquiera de una malla sana)
    rows, cols = 20, 30
    x_a, y_a, z_a, valid_a = _plane_mesh(rows, cols, z_offset=0.0)
    x_b, y_b, z_b, valid_b = _plane_mesh(rows, cols, z_offset=10.0)
    z_b[10, :] = 0.3
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b,
                              window_rows=4, anomaly_frac=0.25)
    assert rep["frac_anomalous"] > 0
    # solo la fila 10 (la del hueco chico real) debe aparecer entre los ejemplos anomalos -- las
    # filas vecinas, con hueco normal y linea base normal, no deben contaminarse
    anomaly_rows = {e["row"] for e in rep["anomaly_examples"]}
    assert anomaly_rows == {10}


def test_neighbor_gap_report_only_dip_row_has_anomalies():
    rows, cols = 20, 30
    x_a, y_a, z_a, valid_a = _plane_mesh(rows, cols, z_offset=0.0)
    x_b, y_b, z_b, valid_b = _plane_mesh(rows, cols, z_offset=10.0)
    z_b[10, :] = 0.3
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b,
                              window_rows=4, anomaly_frac=0.25)
    # 30 columnas, todas en la fila 10 deberian marcarse (hueco 0.3 << 0.25*10)
    assert rep["vertices_anomalous"] == cols
    assert rep["worst_row"]["row"] == 10


def test_neighbor_gap_report_smooth_trend_not_flagged():
    # el hueco crece linealmente con la fila (vueltas mas separadas hacia afuera) -- una linea
    # base LOCAL (ventana de filas vecinas) debe seguir esa tendencia suave sin marcar nada,
    # a diferencia de un umbral global fijo que si marcaria las filas con hueco chico
    rows, cols = 30, 20
    x_a, y_a, z_a, valid_a = _plane_mesh(rows, cols, z_offset=0.0)
    row_idx = np.arange(rows)[:, None]
    x_b, y_b, _, valid_b = _plane_mesh(rows, cols, z_offset=0.0)
    z_b = np.broadcast_to(5.0 + 0.5 * row_idx, (rows, cols)).copy()
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b,
                              window_rows=3, anomaly_frac=0.25)
    assert rep["frac_anomalous"] == 0.0


def test_neighbor_gap_report_insufficient_neighbor_rows_no_baseline():
    # solo 2 filas con datos -- ninguna tiene min_neighbor_rows=2 filas vecinas (excluyendose a
    # si misma), asi que ninguna deberia tener linea base
    x_a, y_a, z_a, valid_a = _plane_mesh(rows=2, cols=10, z_offset=0.0)
    x_b, y_b, z_b, valid_b = _plane_mesh(rows=2, cols=10, z_offset=10.0)
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b,
                              window_rows=3, min_neighbor_rows=2)
    assert rep["n_rows_with_baseline"] == 0
    assert rep["frac_anomalous"] is None


def test_neighbor_gap_report_empty_b_returns_no_data():
    x_a, y_a, z_a, valid_a = _plane_mesh(rows=5, cols=5)
    x_b, y_b, z_b, valid_b = _plane_mesh(rows=5, cols=5)
    valid_b[:] = False
    rep = neighbor_gap_report(x_a, y_a, z_a, valid_a, x_b, y_b, z_b, valid_b)
    assert rep["vertices_measured"] == 0
    assert rep["frac_anomalous"] is None
    assert "error" in rep


def test_check_neighbor_gap_pair_reads_two_dirs(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile

    dir_a, dir_b = tmp_path / "w020", tmp_path / "w021"
    dir_a.mkdir()
    dir_b.mkdir()
    x_a, y_a, z_a, _ = _plane_mesh(rows=15, cols=15, z_offset=1.0)  # z>0 para que cuente como valido
    x_b, y_b, z_b, _ = _plane_mesh(rows=15, cols=15, z_offset=11.0)
    for name, arr in (("x", x_a), ("y", y_a), ("z", z_a)):
        tifffile.imwrite(str(dir_a / f"{name}.tif"), arr.astype(np.float32))
    for name, arr in (("x", x_b), ("y", y_b), ("z", z_b)):
        tifffile.imwrite(str(dir_b / f"{name}.tif"), arr.astype(np.float32))
    rep = check_neighbor_gap_pair(str(dir_a), str(dir_b), window_rows=3)
    assert rep["mesh_dir_a"] == str(dir_a)
    assert rep["mesh_dir_b"] == str(dir_b)
    assert rep["frac_anomalous"] == 0.0
