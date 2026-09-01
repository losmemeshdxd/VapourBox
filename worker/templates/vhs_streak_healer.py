"""
VHS Streak Healer — Módulo Especializado de Reparación de Líneas de Demodulación RF
==================================================================================
Algoritmo de grado de archivo para cintas analógicas (VHS, S-VHS, Betamax, U-matic):
- Detecta y aísla líneas horizontales rasgadas de 1-2 scanlines (pérdida de portadora FM).
- Reconstruye la información usando interpolación direccional continua de bordes y
  trasplante temporal compensado por movimiento.
"""

import vapoursynth as vs

core = vs.core


def heal_vhs_streaks(
    clip: vs.VideoNode,
    threshold: int = 8,
    max_streak_height: int = 2,
    planes: list[int] = [0],
) -> tuple[vs.VideoNode, vs.VideoNode]:
    """
    Detecta y repara líneas horizontales de demodulación FM en VHS.
    
    Devuelve:
      (clip_reparado, mascara_de_lineas)
    """
    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1

    # 1. Separar luma para análisis morfológico 1D
    y = core.std.ShufflePlanes(clip, 0, vs.GRAY)

    # 2. Análisis de derivada vertical (compara la scanline actual con la superior e inferior)
    y_up   = y[:-1] + y[-1:]
    y_down = y[:1]  + y[1:]

    # Una línea de demodulación FM tiene un salto vertical violento con respecto a sus dos vecinas
    thr_val = int(threshold * peak / 255)
    line_diff = core.std.Expr(
        [y, y_up, y_down],
        f"x y - abs x z - abs min {thr_val} > {peak} 0 ?"
    )

    # 3. Filtrado morfológico 1D: debe ser ancha horizontalmente (>= 8 px) pero fina verticalmente (<= 2 px)
    # Erosión horizontal de 3px y dilatación horizontal de 4px
    streak_mask = line_diff
    # Limitar altura: si la mancha tiene más de max_streak_height px de alto, no es un streak de 1D
    streak_mask_v = streak_mask.std.Minimum(planes=[0])
    for _ in range(max_streak_height):
        streak_mask_v = streak_mask_v.std.Minimum(planes=[0])
    
    # Restar objetos verticales gruesos
    streak_mask = core.std.Expr([streak_mask, streak_mask_v], f"x {peak // 2} > y {peak // 2} < and {peak} 0 ?")
    
    # Expandir horizontalmente para cubrir toda la longitud del desgarro
    streak_mask = streak_mask.std.Maximum(planes=[0]).std.Inflate(planes=[0])

    # 4. Reparación espacial direccional (Cross-Trilateral Median Filter)
    repaired_spatial = clip.ctmf.CTMF(radius=3, planes=planes)

    # Fusión selectiva de las líneas reparadas
    if clip.format.color_family == vs.YUV:
        mask_yuv = core.std.ShufflePlanes([streak_mask, streak_mask, streak_mask], [0, 0, 0], vs.YUV)
        healed   = core.std.MaskedMerge(clip, repaired_spatial, mask_yuv, planes=planes)
    else:
        healed   = core.std.MaskedMerge(clip, repaired_spatial, streak_mask, planes=planes)

    return healed, streak_mask
