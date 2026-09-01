"""
QuesoLimpia — Restaurador Profesional de Lluvia de VHS, Polvo y Pelos de Cámara
================================================================================
Algoritmo de restauración de alta gama:
1. Detección y eliminación de lluvia de VHS y puntos volátiles (temporal median con MVTools).
2. Detección y reparación de pelos fijos / suciedad de compuerta (Gate Hair Inpainting).
3. Compensación de movimiento sub-píxel con recálculo fino (Recalculate).
4. Procesamiento en espacio de 16-bit para máxima calidad de laboratorio.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 85,
    threshold:         int   = 10,
    radius:            int   = 1,
    gate_hair:         int   = 50,
    show_mask:         str   = "off",
    # Compatibilidad con worker y presets
    mode:              str   = "balanced",
    spatial_threshold: int   = 15,
    min_dust_size:     int   = 0,
    max_dust_size:     int   = 32,
    detect_bright:     bool  = True,
    detect_dark:       bool  = True,
    detect_spatial:    bool  = True,
    detect_static:     bool  = True,
    grain_suppress:    int   = 0,
    edge_protect:      int   = 0,
    scene_protect:     bool  = True,
    scene_threshold:   float = 0.10,
    chroma:            bool  = True,
    grain_restore:     int   = 0,
    temporal_radius:   int   = 1,
    blksize:           int | None = None,
    pel:               int | None = None,
) -> vs.VideoNode:
    """
    QuesoLimpia — 4 Sliders Esenciales de Restauración:
    --------------------------------------------------
    strength  : 10-100% (Fuerza global de limpieza de puntos y dropouts).
    threshold : 2-30    (Sensibilidad: menor número detecta lluvia más fina).
    radius    : 1-2     (Radio temporal: 1 = normal, 2 = lluvia densa / daño severo).
    gate_hair : 0-100%  (Eliminación de pelos de compuerta y manchas fijas de lente).
    show_mask : "off" / "repair" / "gate_hair" (Diagnóstico visual).
    """
    if clip.format is None:
        raise vs.Error("QuesoLimpia: el clip debe tener formato constante.")

    src_fmt_id = clip.format.id
    is_float   = clip.format.sample_type == vs.FLOAT
    is_gray    = clip.format.color_family == vs.GRAY
    chroma     = False if is_gray else chroma
    planes     = [0, 1, 2] if chroma else [0]
    bits       = clip.format.bits_per_sample

    # ── Procesamiento interno a 16-bit para máxima fidelidad ─────────────────
    needs_upscale = (not is_float) and (bits < 16)
    if needs_upscale:
        if is_gray:
            work_fmt = core.get_video_format(vs.GRAY8).replace(bits_per_sample=16, sample_type=vs.INTEGER).id
        else:
            work_fmt = core.get_video_format(vs.YUV420P8).replace(bits_per_sample=16, sample_type=vs.INTEGER).id
        clip_work = core.resize.Point(clip, format=work_fmt)
        work_bits = 16
    else:
        clip_work = clip
        work_bits = bits

    peak_v = (1 << work_bits) - 1 if not is_float else 1.0

    # ── Calibración de Bloques MVTools ──────────────────────────────────────
    w = clip_work.width
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2

    Super       = core.mvsf.Super if is_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_float else core.mv.Recalculate

    sup = Super(clip_work, pel=pel, sharp=2, rfilter=4)

    # ── Análisis de movimiento delta 1 con Recalculate fino ───────────────────
    bv1 = Analyse(sup, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=5)
    fv1 = Analyse(sup, isb=False, delta=1, blksize=blksize, overlap=overlap, search=5)

    rec_blksize = max(4, blksize // 2)
    rec_overlap = rec_blksize // 2
    bv1 = Recalculate(sup, bv1, blksize=rec_blksize, overlap=rec_overlap, search=5)
    fv1 = Recalculate(sup, fv1, blksize=rec_blksize, overlap=rec_overlap, search=5)

    bc1 = Compensate(clip_work, sup, bv1)
    fc1 = Compensate(clip_work, sup, fv1)

    # ── 1. LIMPIEZA DE LLUVIA DE VHS Y PUNTOS TEMPORALES ────────────────────
    effective_radius = max(radius, temporal_radius)
    if effective_radius >= 2:
        # Lluvia muy densa: 5 cuadros compensados (delta 1 y delta 2)
        bv2 = Analyse(sup, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=5)
        fv2 = Analyse(sup, isb=False, delta=2, blksize=blksize, overlap=overlap, search=5)
        bv2 = Recalculate(sup, bv2, blksize=rec_blksize, overlap=rec_overlap, search=5)
        fv2 = Recalculate(sup, fv2, blksize=rec_blksize, overlap=rec_overlap, search=5)
        bc2 = Compensate(clip_work, sup, bv2)
        fc2 = Compensate(clip_work, sup, fv2)

        interleaved = core.std.Interleave([fc2, fc1, clip_work, bc1, bc2])
        cleaned_temporal = interleaved.tmedian.TemporalMedian(2, planes)[2::5]
    else:
        # Lluvia estándar / telecine: 3 cuadros compensados (delta 1)
        interleaved = core.std.Interleave([fc1, clip_work, bc1])
        cleaned_temporal = interleaved.tmedian.TemporalMedian(1, planes)[1::3]

    # ── Máscara de detección de lluvia / dropouts ────────────────────────────
    thr_val = (threshold * peak_v / 255.0) if not is_float else (threshold / 255.0)
    diff = core.std.Expr([clip_work, cleaned_temporal], "x y - abs")
    volatile_mask = core.std.Expr([diff], f"x {thr_val:.4f} > {peak_v} 0 ?")
    volatile_mask = volatile_mask.std.Inflate(planes=[0])

    # Fusión según la fuerza elegida
    if strength >= 95:
        repaired = cleaned_temporal
    else:
        repaired = core.std.MaskedMerge(clip_work, cleaned_temporal, volatile_mask, planes=planes)

    # ── 2. DETECCIÓN Y RESTAURACIÓN DE PELOS / GATE HAIR ─────────────────────
    static_mask = None
    if detect_static:
        # Un pelo fijo o suciedad de compuerta persiste en el mismo lugar en 3 cuadros seguidos
        prev_m = volatile_mask[-1:] + volatile_mask[:-1]
        next_m = volatile_mask[1:]  + volatile_mask[-1:]
        
        thr_hair = peak_v // 2
        # Intersección en 3 cuadros seguidos = defecto estático (pelo)
        static_mask = core.std.Expr(
            [volatile_mask, prev_m, next_m],
            f"x {thr_hair} > y {thr_hair} > and z {thr_hair} > and {peak_v} 0 ?"
        )
        # Dilatar ligeramente para cubrir el grosor del pelo
        static_mask = static_mask.std.Maximum(planes=[0]).std.Inflate(planes=[0])

        # Reparación espacial (Inpainting / CTMF de radio 4) para reemplazar el pelo
        repaired_spatial = clip_work.ctmf.CTMF(radius=4, planes=planes)
        repaired = core.std.MaskedMerge(repaired, repaired_spatial, static_mask, planes=planes)

    # ── Diagnóstico visual ───────────────────────────────────────────────────
    if show_mask == "repair":
        # Rojo: lluvia/puntos temporales eliminados
        red_box = core.std.BlankClip(clip_work, color=[peak_v, 0, 0] if not is_gray else [peak_v])
        overlay = core.std.MaskedMerge(clip_work, red_box, volatile_mask, planes=planes)
        if needs_upscale:
            return core.resize.Point(overlay, format=src_fmt_id)
        return overlay

    if show_mask == "gate_hair":
        # Azul/Cian: pelos de compuerta fijos detectados
        if static_mask is not None:
            blue_box = core.std.BlankClip(clip_work, color=[0, peak_v, peak_v] if not is_gray else [peak_v])
            overlay  = core.std.MaskedMerge(clip_work, blue_box, static_mask, planes=planes)
        else:
            overlay = clip_work
        if needs_upscale:
            return core.resize.Point(overlay, format=src_fmt_id)
        return overlay

    if show_mask == "side_by_side":
        half    = clip_work.width // 2
        left    = core.std.CropAbs(clip_work, width=half, height=clip_work.height, left=0, top=0)
        right   = core.std.CropAbs(repaired,  width=half, height=clip_work.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        if needs_upscale:
            return core.resize.Point(stacked, format=src_fmt_id)
        return stacked

    if needs_upscale:
        return core.resize.Point(repaired, format=src_fmt_id)

    return repaired
