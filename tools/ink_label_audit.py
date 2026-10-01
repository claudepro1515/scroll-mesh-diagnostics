"""Auditoria de etiquetas de tinta publicadas (Frente G, candidata 3).

Motivacion: el reto premio en julio 2026 un "arnes de validacion de tinta" (herramienta chica,
1,000 USD) -- una auditoria reusable de las etiquetas de tinta YA PUBLICADAS (sin modelo, sin
GPU, sin CT) encaja en esa misma categoria y es 100% offline salvo por leer datos publicos.

Censo previo (27 sep 2026, turno automatico, ver wm/MESH_TOPOLOGY_CHECK.md y Bitacora): de los
39 segmentos curados de PHerc0139, solo 8 tienen `ink-labels/` publicado en el bucket S3 oficial
(`<segmento>/ink-labels/<res>um-volume-<fecha>/<fecha-publicacion>/{inklabels,supervision}.zarr`,
zarr v3 sharding_indexed+blosc, un solo `sharding` por nivel -- tool/shard_v3.py ya sabe leerlo).
Cada combinacion segmento+volumen solo tiene UNA fecha de publicacion (20260918): no hay
revisiones que comparar en el tiempo. 4 de los 8 segmentos tienen etiquetas publicadas en DOS
resoluciones distintas (2.399um y 9.362um) para el mismo segmento fisico -- eso si da una
auditoria real posible: comparar las dos etiquetas INDEPENDIENTES del mismo trozo de papiro.

Dos chequeos, ambos sobre datos ya publicados (nunca inventamos verdad, solo consistencia interna):

1. `label_supervision_consistency(labels, supervision)`: la mascara "supervision" marca la zona
   que un anotador de verdad revizo (donde una ausencia de tinta es una afirmacion real, no solo
   "no se sabe"). Un pixel de `labels=True` FUERA de `supervision` no tiene sentido -- el
   anotador no pudo confirmar tinta en una zona que el mismo no reviso. Esto es un chequeo de
   consistencia interna del PAR de arrays (no necesita un segundo dataset ni un modelo), analogo
   en espiritu a mesh_topology_check.py (pliegues de una malla contra si misma). Tambien marca
   degeneracion (supervision vacia o llena, sin nada medible) y desfase de forma (bug real
   plausible: subir la mascara equivocada).

2. `cross_resolution_agreement(...)`: cuando el MISMO segmento tiene etiquetas publicadas a dos
   resoluciones distintas, ambas cubren aproximadamente la misma extension fisica del papiro (el
   mismo tifxyz aplanado) aunque con distinto numero de pixeles -- as que la FRACCION de tinta
   dentro de la zona supervisada deberia ser parecida entre ambas, sin necesidad de alinear pixel
   a pixel (evita el problema mas dificil de registro geometrico). Una discrepancia grande entre
   las dos fracciones es evidencia real de un problema (mal registro entre las dos publicaciones,
   o un error de anotacion en una de ellas) -- justo el tipo de hallazgo que un "arnes de
   validacion de tinta" deberia sacar a la luz.

Todo lo que toca la red (fetch de S3) vive en funciones separadas (`fetch_*`) de las funciones
puras de analisis (`label_supervision_consistency`, `cross_resolution_agreement`) para poder
probar estas ultimas offline con arrays sinteticos, sin depender de la red -- ver
wm/tests/test_ink_label_audit.py (regla del sistema: probar con sinteticos ANTES de datos reales).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tool")
if TOOL not in sys.path:
    sys.path.insert(0, TOOL)

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


# ---------------------------------------------------------------------------
# Analisis puro (sin red): funciones que reciben arrays ya cargados
# ---------------------------------------------------------------------------

def spillover_locality(labels: np.ndarray, supervision: np.ndarray, pad: int = 60) -> dict:
    """De la tinta que cae FUERA de la zona supervisada, cuanta es "bleed" normal de borde
    (a <= `pad` pixeles del area supervisada -- un trazo de pincel que se pasa un poco del
    poligono dibujado, esperable) contra cuanta esta REALMENTE lejos (mas alla del margen, o
    directamente fuera de la caja del area supervisada expandida) -- eso ultimo no se explica
    por bleed de borde, es una region de tinta separada sin ninguna revision de un anotador cerca.

    Para no gastar memoria en arrays gigantes (los de 2.399 um llegan a 33280x30000 = ~1e9 px),
    la transformada de distancia solo se calcula en un recorte alrededor de la caja de la zona
    supervisada (+pad); cualquier pixel de tinta "afuera" que caiga fuera de ese recorte ya es,
    por construccion, mas lejano que `pad` -- se cuenta como lejano sin calcular su distancia exacta.
    """
    ys, xs = np.where(supervision)
    if ys.size == 0:
        return {"ok": True, "n_outside": int((labels & ~supervision).sum()), "n_far": None,
                "frac_far_of_outside": None, "note": "supervision vacia, no aplica"}
    y0, y1 = max(0, ys.min() - pad), min(supervision.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(supervision.shape[1], xs.max() + pad + 1)

    outside_full = labels & ~supervision
    n_outside = int(outside_full.sum())
    if n_outside == 0:
        return {"ok": True, "n_outside": 0, "n_far": 0, "frac_far_of_outside": 0.0}

    # todo lo que cae fuera del recorte (caja de supervision + pad) es, por construccion, lejano
    n_far_outside_crop = n_outside - int(outside_full[y0:y1, x0:x1].sum())

    from scipy import ndimage
    sup_crop = supervision[y0:y1, x0:x1]
    out_crop = outside_full[y0:y1, x0:x1]
    # limite duro de tamano para el crop en el que se corre la transformada de distancia (los
    # segmentos a 2.399um pueden tener una zona supervisada que abarca miles de pixeles en cada
    # eje) -- por encima de MAX_EDT_SIDE por lado, remuestrear con max-pooling antes del EDT
    # (conserva la forma real de la mascara, solo pierde precision sub-bloque) para que esto
    # nunca tome minutos ni GB de RAM sin importar el tamano real del segmento
    # nunca se materializa un array de distancia a resolucion completa: para f>1 se consulta
    # dist_small solo en las coordenadas (remuestreadas) de los pixeles "afuera" que ya existen
    # (tipicamente muchisimo menos que el crop completo), nunca se reconstruye el crop entero
    MAX_EDT_SIDE = 3000
    f = max(1, -(-max(sup_crop.shape) // MAX_EDT_SIDE))  # ceil division
    if f > 1:
        h, w = sup_crop.shape
        h2, w2 = -(-h // f) * f, -(-w // f) * f  # redondear hacia arriba a multiplo de f
        sup_pad = np.zeros((h2, w2), dtype=bool)
        sup_pad[:h, :w] = sup_crop
        sup_small = sup_pad.reshape(h2 // f, f, w2 // f, f).max(axis=(1, 3))
        dist_small = ndimage.distance_transform_edt(~sup_small) * f  # vuelve a unidades de pixel real
        oy, ox = np.where(out_crop)
        n_far_in_crop = int((dist_small[oy // f, ox // f] > pad).sum()) if oy.size else 0
    else:
        dist = ndimage.distance_transform_edt(~sup_crop)
        n_far_in_crop = int((out_crop & (dist > pad)).sum())

    n_far = n_far_outside_crop + n_far_in_crop
    return {"ok": True, "n_outside": n_outside, "n_far": n_far,
            "frac_far_of_outside": n_far / n_outside, "pad_px": pad}


def label_supervision_consistency(labels: np.ndarray, supervision: np.ndarray, pad: int = 60) -> dict:
    """Consistencia interna de un par (inklabels, supervision) ya cargado.

    labels/supervision: arrays 2D booleanos (o que se puedan comparar > 0), misma unidad fisica.
    Devuelve estadisticas y banderas; nunca lanza excepcion por datos "malos" (eso es justo lo
    que se quiere reportar), solo por forma incompatible entre los dos arrays (eso si es un bug
    de verdad -- subieron el archivo equivocado o el segmento equivocado).
    """
    labels = np.asarray(labels).astype(bool)
    supervision = np.asarray(supervision).astype(bool)
    if labels.shape != supervision.shape:
        return {
            "ok": False,
            "error": f"shape mismatch: labels {labels.shape} vs supervision {supervision.shape}",
        }

    n = labels.size
    frac_supervised = float(supervision.mean()) if n else 0.0
    n_sup = int(supervision.sum())
    n_ink = int(labels.sum())
    n_ink_outside_sup = int((labels & ~supervision).sum())
    frac_ink_outside_sup = (n_ink_outside_sup / n_ink) if n_ink else 0.0
    frac_ink_within_sup = (float((labels & supervision).sum()) / n_sup) if n_sup else 0.0
    locality = spillover_locality(labels, supervision, pad=pad) if n_ink_outside_sup else {
        "ok": True, "n_outside": 0, "n_far": 0, "frac_far_of_outside": 0.0}

    flags = []
    # bleed de borde (tinta a <= pad px del area revisada) es normal; lo que se marca es tinta
    # LEJOS de cualquier zona revisada -- solo eso es un hallazgo real, no ruido de anotacion
    if locality.get("n_far") and locality["n_far"] > 500 and (locality.get("frac_far_of_outside") or 0) > 0.2:
        flags.append("ink_far_outside_supervision")
    elif n_ink_outside_sup > 0:
        flags.append("labels_outside_supervision_bleed")  # normal, informativo, no es hallazgo
    if frac_supervised == 0.0:
        flags.append("empty_supervision")
    elif frac_supervised == 1.0:
        flags.append("full_supervision")  # sospechoso: ninguna zona quedo fuera de revision
    if n_ink == 0:
        flags.append("no_ink_labelled")  # no es necesariamente un bug (puede ser un control sin tinta)

    return {
        "ok": True,
        "shape": list(labels.shape),
        "frac_supervised": frac_supervised,
        "n_ink_px": n_ink,
        "spillover_locality": locality,
        "n_ink_outside_supervision_px": n_ink_outside_sup,
        "frac_ink_outside_supervision": frac_ink_outside_sup,
        "frac_ink_within_supervision": frac_ink_within_sup,
        "flags": flags,
    }


def cross_resolution_agreement(stats_a: dict, stats_b: dict, tol: float = 0.35) -> dict:
    """Compara la fraccion de tinta DENTRO de la zona supervisada entre dos publicaciones
    independientes del mismo segmento a dos resoluciones distintas (misma extension fisica,
    distinto muestreo de pixeles). No compara pixel a pixel (eso pediria alinear las dos mallas
    aplanadas, un problema mas dificil y no necesario para esta auditoria).

    stats_a/stats_b: el dict que devuelve label_supervision_consistency para cada resolucion.
    tol: diferencia ABSOLUTA maxima tolerada entre las dos fracciones antes de marcar bandera
    (0.35 es deliberadamente holgado -- dos anotaciones independientes de distinta resolucion no
    van a coincidir en pixeles exactos ni siquiera cuando ambas son correctas; el chequeo busca
    discrepancias GRANDES, tipo "una dice 80% de tinta y la otra 5%", no ruido normal de anotacion).
    """
    if not stats_a.get("ok") or not stats_b.get("ok"):
        return {"ok": False, "error": "una de las dos entradas no es valida (ver shape mismatch)"}
    fa = stats_a["frac_ink_within_supervision"]
    fb = stats_b["frac_ink_within_supervision"]
    diff = abs(fa - fb)
    return {
        "ok": True,
        "frac_ink_within_supervision_a": fa,
        "frac_ink_within_supervision_b": fb,
        "abs_diff": diff,
        "flags": ["cross_resolution_disagreement"] if diff > tol else [],
    }


# ---------------------------------------------------------------------------
# Red: descubrir y leer los datos reales publicados
# ---------------------------------------------------------------------------

def list_ink_label_sets(segment_prefix: str) -> list[dict]:
    """Para un segmento (p.ej. 'PHerc0139/segments/20260317000000-w035_2026031718/'), lista
    cada combinacion volumen+fecha-de-publicacion de ink-labels que exista, con su URL base."""
    from vcz import list_prefixes  # import diferido: solo hace falta con red

    out = []
    for vol_prefix in list_prefixes(segment_prefix + "ink-labels/"):
        volume = vol_prefix.rstrip("/").split("/")[-1]
        for pub_prefix in list_prefixes(vol_prefix):
            pub_date = pub_prefix.rstrip("/").split("/")[-1]
            out.append({"segment_prefix": segment_prefix, "volume": volume, "pub_date": pub_date,
                        "base": f"{BUCKET}/{pub_prefix.rstrip('/')}"})
    return out


def voxel_um_from_volume_key(volume: str) -> float:
    """'2.399um-volume-20260102150214' -> 2.399. El pad de spillover_locality tiene que
    convertirse a pixeles POR RESOLUCION (nunca un mismo numero de pixeles para las dos): un
    pad de 60 px es una tolerancia fisica ~3.9x mas chica a 2.399 um que a 9.362 um -- usar el
    mismo pad en pixeles en las dos resoluciones habria hecho que la MISMA tinta se marcara
    "lejana" en una resolucion y "bleed normal" en la otra, sin que el papiro haya cambiado
    (visto de verdad: w030/w040/w045 marcaban distinto segun la resolucion antes de este arreglo)."""
    import re
    m = re.match(r"([0-9.]+)um", volume)
    if not m:
        raise ValueError(f"no se pudo leer el tamano de voxel de {volume!r}")
    return float(m.group(1))


def fetch_label_set(base_url: str, volume: str, pad_um: float = 560.0) -> dict:
    """Descarga inklabels.zarr/0 y supervision.zarr/0 (nivel 0 = resolucion completa de esa
    publicacion; para estos segmentos ya son crops chicos, unos cientos de KB) y corre el
    chequeo de consistencia interna, con el pad de bleed convertido a pixeles de ESTA resolucion
    a partir de una tolerancia fisica fija en micras (pad_um), para que el chequeo sea comparable
    entre publicaciones a distinta resolucion del mismo segmento."""
    from shard_v3 import read_sharded_2d

    labels = read_sharded_2d(f"{base_url}/inklabels.zarr/0") > 0
    supervision = read_sharded_2d(f"{base_url}/supervision.zarr/0") > 0
    pad_px = max(1, round(pad_um / voxel_um_from_volume_key(volume)))
    return label_supervision_consistency(labels, supervision, pad=pad_px)


def audit_segment(segment_prefix: str, pad_um: float = 560.0) -> dict:
    """Corre el chequeo 1 en cada resolucion publicada de un segmento, y el chequeo 2 (cruce
    entre resoluciones) si hay mas de una."""
    sets = list_ink_label_sets(segment_prefix)
    per_volume = {}
    for s in sets:
        key = f'{s["volume"]}@{s["pub_date"]}'
        per_volume[key] = {**s, "stats": fetch_label_set(s["base"], s["volume"], pad_um=pad_um)}

    result = {"segment": segment_prefix, "n_volumes": len(per_volume), "per_volume": per_volume,
              "cross_resolution": []}
    keys = list(per_volume.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cr = cross_resolution_agreement(per_volume[keys[i]]["stats"], per_volume[keys[j]]["stats"])
            cr["pair"] = [keys[i], keys[j]]
            result["cross_resolution"].append(cr)
    return result


DEFAULT_SEGMENTS = [
    "PHerc0139/segments/20250108000005-w030_2025010818/",
    "PHerc0139/segments/20250831000000-w040_2025083102/",
    "PHerc0139/segments/20260108000000-w041_2026010816/",
    "PHerc0139/segments/20260112000000-w043_2026011217/",
    "PHerc0139/segments/20260115000000-w044_2026011522/",
    "PHerc0139/segments/20260126000000-w045_2026012619/",
    "PHerc0139/segments/20260302000000-w039_2026030210/",
    "PHerc0139/segments/20260317000000-w035_2026031718/",
]


def main():
    segs = sys.argv[1:] or DEFAULT_SEGMENTS
    report = []
    for s in segs:
        seg = audit_segment(s)
        report.append(seg)
        name = seg["segment"].rstrip("/").split("/")[-1]
        for key, v in seg["per_volume"].items():
            st = v["stats"]
            flags = ",".join(st.get("flags", [])) or "-"
            print(f"{name} {key}: frac_ink_in_sup={st.get('frac_ink_within_supervision', 0):.3f} "
                  f"flags=[{flags}]", flush=True)
        for cr in seg["cross_resolution"]:
            flags = ",".join(cr.get("flags", [])) or "-"
            print(f"  {name} cruce {cr['pair'][0]} vs {cr['pair'][1]}: "
                  f"diff={cr.get('abs_diff', float('nan')):.3f} flags=[{flags}]", flush=True)
    out_path = os.path.join(HERE, "data_topology", "ink_label_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"guardado: {out_path}")


if __name__ == "__main__":
    main()
