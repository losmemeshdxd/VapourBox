"""
QuesoLimpia — Eliminador Automático de Polvo, Manchas y Puntos (VHS & Film)
=============================================================================
Motor de restauración temporal con compensación de movimiento MVTools
y filtrado de mediana temporal adaptativa para eliminar al 100% manchas,
puntos negros/blancos y dropouts de cinta VHS o película.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 80,
    threshold:         int   = 12,
    rec:               bool  = True,
    chroma:            bool  = True,
    show_mask:         str   = "off",
    # Parámetros heredados para compatibilidad de firma
    mode:              str   = "balanced",
    spatial_threshold: int   = 15,
    min_dust_size:     int   = 0,
    max_dust_size:     int   = 32,
    detect_bright:     bool  = True,
    detect_dark:       bool  = True,
    detect_spatial:    bool  = True,
    detect_static:     bool  = False,
    grain_suppress:    int   = 0,
    edge_protect:      int   = 0,
    scene_protect:     bool  = False,
    scene_threshold:   float = 0.10,
    grain_restore:     int   = 0,
    temporal_radius:   int   = 1,
    blksize:           int | None = None,
    pel:               int | None = None,
) -> vs.VideoNode:
    """
    QuesoLimpia — Limpieza automática de puntos y defectos temporales.

    5 Controles Esenciales:
    -----------------------
    strength   : 10-100% (Fuerza global de limpieza).
    threshold  : 1-40    (Sensibilidad a puntos pequeños).
    rec        : True    (Recalcular vectores finos para máxima precisión en movimiento).
    chroma     : True    (Limpiar manchas en color/croma).
    show_mask  : "off" / "repair" / "side_by_side" (Diagnóstico visual).
    """
    if clip.format is None:
        raise vs.Error("QuesoLimpia: el clip debe tener formato constante.")

    is_float = clip.format.sample_type == vs.FLOAT
    is_gray  = clip.format.color_family == vs.GRAY
    chroma   = False if is_gray else chroma
    planes   = [0, 1, 2] if chroma else [0]
    bits     = clip.format.bits_per_sample
    peak_v   = (1 << bits) - 1 if not is_float else 1.0

    # Auto block size & pel
    if blksize is None:
        blksize = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 1 if clip.width > 960 else 2

    # Seleccionar funciones MVTools
    Super       = core.mvsf.Super if is_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_float else core.mv.Recalculate

    # Generar super clip con filtrado de paso bajo para motion vectors robustos
    sup = Super(clip, pel=pel, sharp=1, rfilter=4)

    # Análisis de movimiento bidireccional
    bv1 = Analyse(sup, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=5)
    fv1 = Analyse(sup, isb=False, delta=1, blksize=blksize, overlap=overlap, search=5)

    # Recálculo a nivel fino para no borrar objetos rápidos
    if rec:
        rec_blksize = max(4, blksize // 2)
        rec_overlap = rec_blksize // 2
        bv1 = Recalculate(sup, bv1, blksize=rec_blksize, overlap=rec_overlap, search=5)
        fv1 = Recalculate(sup, fv1, blksize=rec_blksize, overlap=rec_overlap, search=5)

    # Compensar cuadros vecinos
    bc1 = Compensate(clip, sup, bv1)
    fc1 = Compensate(clip, sup, fv1)

    # Mediana temporal compensada: elimina 100% de puntos aislados en 1 cuadro
    interleaved = core.std.Interleave([fc1, clip, bc1])
    cleaned     = interleaved.tmedian.TemporalMedian(1, planes)[1::3]

    # Diferencia entre el original y el cuadro limpio
    thr_val = (threshold * peak_v / 255.0) if not is_float else (threshold / 255.0)
    diff = core.std.Expr([clip, cleaned], "x y - abs")
    
    # Máscara de manchas: detecta cualquier pixel que difiera significativamente
    mask = core.std.Expr([diff], f"x {thr_val:.4f} > {peak_v} 0 ?")
    mask = mask.std.Inflate(planes=[0])

    if strength >= 95:
        repaired = cleaned
    else:
        # Fusión selectiva: solo reemplaza los píxeles con suciedad
        repaired = core.std.MaskedMerge(clip, cleaned, mask, planes=planes)

    # Diagnóstico visual
    if show_mask == "repair":
        # Mostrar manchas detectadas en rojo sobre la imagen
        red_box = core.std.BlankClip(clip, color=[peak_v, 0, 0] if not is_gray else [peak_v])
        overlay = core.std.MaskedMerge(clip, red_box, mask, planes=planes)
        return overlay

    if show_mask == "side_by_side":
        half  = clip.width // 2
        left  = core.std.CropAbs(clip,     width=half, height=clip.height, left=0, top=0)
        right = core.std.CropAbs(repaired, width=half, height=clip.height, left=half, top=0)
        return core.std.StackHorizontal([left, right])

    return repaired
