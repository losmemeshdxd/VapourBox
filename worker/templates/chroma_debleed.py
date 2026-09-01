"""
Chroma De-Bleed & Comet-Tail Suppressor — Módulo Especializado de Croma VHS
===========================================================================
Elimina el sangrado de color, ruido de portadora Color-Under (629 kHz) y
colas de cometa cromáticas (comet-tailing) características de cintas analógicas.
"""

import vapoursynth as vs

core = vs.core


def restore_vhs_chroma(
    clip: vs.VideoNode,
    chroma_strength: int = 80,
    chroma_threshold: int = 25,
) -> vs.VideoNode:
    """
    Restaura y desmancha los planos de croma (U y V) en cintas VHS.
    """
    if clip.format.color_family != vs.YUV:
        return clip

    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1

    # Separar planos Y, U, V
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    u = core.std.ShufflePlanes(clip, 1, vs.GRAY)
    v = core.std.ShufflePlanes(clip, 2, vs.GRAY)

    # 1. Mediana espacial adaptativa en croma para erradicar ruido de alta frecuencia
    u_clean = u.ctmf.CTMF(radius=3)
    v_clean = v.ctmf.CTMF(radius=3)

    # 2. Detección de manchas y colas de cometa en croma
    thr_c = int(chroma_threshold * peak / 255)
    diff_u = core.std.Expr([u, u_clean], f"x y - abs {thr_c} > {peak} 0 ?").std.Inflate()
    diff_v = core.std.Expr([v, v_clean], f"x y - abs {thr_c} > {peak} 0 ?").std.Inflate()

    # 3. Fusión selectiva de croma
    if chroma_strength >= 95:
        u_final = u_clean
        v_final = v_clean
    else:
        weight = chroma_strength / 100.0
        u_weighted = core.std.Expr([diff_u], f"x {weight:.3f} *")
        v_weighted = core.std.Expr([diff_v], f"x {weight:.3f} *")
        u_final = core.std.MaskedMerge(u, u_clean, u_weighted)
        v_final = core.std.MaskedMerge(v, v_clean, v_weighted)

    # Recombinar en clip YUV intacto
    return core.std.ShufflePlanes([y, u_final, v_final], [0, 0, 0], vs.YUV)
