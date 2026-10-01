import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ink_label_audit import (  # noqa: E402
    label_supervision_consistency, cross_resolution_agreement, spillover_locality,
    voxel_um_from_volume_key,
)


def _tile(h=20, w=20):
    return np.zeros((h, w), dtype=bool)


def test_clean_labels_within_supervision_no_flags():
    sup = _tile()
    sup[2:18, 2:18] = True  # zona revisada por el anotador
    lab = _tile()
    lab[5:10, 5:10] = True  # tinta, toda dentro de la zona revisada
    st = label_supervision_consistency(lab, sup)
    assert st["ok"]
    assert st["flags"] == []
    assert st["n_ink_outside_supervision_px"] == 0
    assert np.isclose(st["frac_ink_within_supervision"], lab.sum() / sup.sum())


def test_small_border_bleed_is_not_a_real_finding():
    sup = _tile()
    sup[2:18, 2:18] = True
    lab = _tile()
    lab[0:2, 0:2] = True  # tinta FUERA de la zona revisada, pero PEGADA al borde (bleed tipico)
    st = label_supervision_consistency(lab, sup)
    assert st["ok"]
    assert "labels_outside_supervision_bleed" in st["flags"]
    assert "ink_far_outside_supervision" not in st["flags"]
    assert st["n_ink_outside_supervision_px"] == 4
    assert np.isclose(st["frac_ink_outside_supervision"], 1.0)  # toda la tinta esta afuera


def test_large_far_blob_is_flagged_as_real_finding():
    # zona supervisada chica en una esquina; un blob grande de tinta bien lejos (no es bleed de
    # borde: ni su tamano ni su distancia se explican por un trazo que se paso un poco del poligono)
    sup = np.zeros((400, 400), dtype=bool)
    sup[10:30, 10:30] = True  # 20x20 supervisado
    lab = np.zeros((400, 400), dtype=bool)
    lab[15:20, 15:20] = True  # un poco de tinta normal, dentro de lo supervisado
    lab[300:340, 300:340] = True  # blob de 1600 px, a mas de 250 px del area supervisada
    st = label_supervision_consistency(lab, sup, pad=15)
    assert st["ok"]
    assert "ink_far_outside_supervision" in st["flags"]
    loc = st["spillover_locality"]
    assert loc["n_far"] >= 1500  # casi todo el blob lejano cuenta como "lejano"


def test_large_far_blob_flagged_even_with_downsample_safety_cap():
    # crop mayor al limite duro de spillover_locality (MAX_EDT_SIDE=3000) -- fuerza la ruta de
    # max-pooling antes del EDT; debe seguir distinguiendo bleed (cerca) de un blob real (lejos)
    n = 3400
    sup = np.zeros((n, n), dtype=bool)
    sup[100:3300, 100:3300] = True  # zona supervisada grande, bbox > 3000 px de lado
    lab = np.zeros((n, n), dtype=bool)
    lab[100:110, 100:110] = True  # tinta normal, dentro de lo supervisado
    lab[97:99, 97:99] = True  # 2-3 px de bleed pegados al borde
    st = label_supervision_consistency(lab, sup, pad=60)
    assert st["ok"]
    loc = st["spillover_locality"]
    assert loc["n_far"] == 0  # el bleed de 2-3 px no cuenta como lejano
    assert "ink_far_outside_supervision" not in st["flags"]


def test_empty_supervision_is_flagged_degenerate():
    sup = _tile()  # nada revisado
    lab = _tile()
    st = label_supervision_consistency(lab, sup)
    assert "empty_supervision" in st["flags"]
    assert "no_ink_labelled" in st["flags"]
    assert st["frac_ink_within_supervision"] == 0.0  # division por cero evitada, no NaN


def test_full_supervision_is_flagged_suspicious():
    sup = np.ones((10, 10), dtype=bool)  # todo el tile "revisado", sospechoso en un crop real
    lab = _tile(10, 10)
    lab[3:5, 3:5] = True
    st = label_supervision_consistency(lab, sup)
    assert "full_supervision" in st["flags"]
    assert "ink_far_outside_supervision" not in st["flags"]
    assert "labels_outside_supervision_bleed" not in st["flags"]


def test_shape_mismatch_is_reported_not_raised():
    sup = _tile(20, 20)
    lab = _tile(10, 10)
    st = label_supervision_consistency(lab, sup)
    assert st["ok"] is False
    assert "shape mismatch" in st["error"]


def test_cross_resolution_agreement_flags_large_gap_not_small():
    sup = _tile(30, 30)
    sup[:, :] = True
    lab_a = _tile(30, 30)
    lab_a[:9, :] = True  # 30% de tinta
    lab_b = _tile(30, 30)
    lab_b[:12, :] = True  # 40% de tinta (distinta resolucion, misma zona fisica aprox.)
    st_a = label_supervision_consistency(lab_a, sup)
    st_b = label_supervision_consistency(lab_b, sup)
    close = cross_resolution_agreement(st_a, st_b)
    assert close["flags"] == []  # 10 puntos de diferencia, dentro de tolerancia (ruido normal)

    lab_c = _tile(30, 30)
    lab_c[:27, :] = True  # 90% de tinta: misma zona fisica, pero MUY distinto -> desacuerdo real
    st_c = label_supervision_consistency(lab_c, sup)
    far = cross_resolution_agreement(st_a, st_c)
    assert "cross_resolution_disagreement" in far["flags"]


def test_voxel_um_from_volume_key_parses_leading_float():
    assert voxel_um_from_volume_key("2.399um-volume-20260102150214") == 2.399
    assert voxel_um_from_volume_key("9.362um-volume-20250728140407") == 9.362


def test_spillover_locality_direct_empty_supervision():
    sup = np.zeros((20, 20), dtype=bool)
    lab = np.zeros((20, 20), dtype=bool)
    lab[0:2, 0:2] = True
    loc = spillover_locality(lab, sup)
    assert loc["ok"]
    assert loc["n_far"] is None  # no hay zona supervisada de referencia, no aplica


def test_cross_resolution_agreement_propagates_invalid_input():
    sup = _tile(20, 20)
    lab_ok = _tile(20, 20)
    lab_bad = _tile(5, 5)
    st_ok = label_supervision_consistency(lab_ok, sup)
    st_bad = label_supervision_consistency(lab_bad, sup)  # shape mismatch -> ok: False
    res = cross_resolution_agreement(st_ok, st_bad)
    assert res["ok"] is False
