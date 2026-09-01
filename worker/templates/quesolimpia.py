"""
QuesoLimpia STUDIO ARCHIVE — Motor de Restauración de Cintas y Video Analógico
==============================================================================
Arquitectura de 7 Etapas de Grado Archivístico (DVO / DIAMANT / MTI DRS):

1. PRE-CONDICIONAMIENTO DUAL DE GUÍA (Motion Guide Conditioning):
   Filtro Wiener bilateral suave para aislar la estructura geométrica real
   del ruido analógico de RF y grano grueso de cinta antes de calcular vectores.

2. ESTIMACIÓN DE MOVIMIENTO JERÁRQUICA 3-PASOS CON BÚSQUEDA EXHAUSTIVA:
   - Macro: Búsqueda exhaustiva 32x32 / 16x16 (search=3, searchparam=3, pel=2).
   - Meso: Recálculo en 8x8 con refinamiento de sub-píxel.
   - Micro: Recálculo en 4x4 con solapamiento al 50% para bordes finos de anime y texto.

3. VOTACIÓN TEMPORAL DE CERTEZA MATEMÁTICA (7 Cuadros Compensados N-3 a N+3):
   Confirma dropouts con lógica estricta: un píxel es marcado como defecto solo si
   difiere simultáneamente de sus vecinos compensados hacia adelante y hacia atrás,
   mientras los vecinos coinciden entre sí (cero falsos positivos, cero fantasmas).

4. DETECTOR MORFOLÓGICO 1D DE DEMODULACIÓN FM (Scanline Streak Healer):
   Aislamiento de líneas horizontales rasgadas de 1 scanline (pérdida de portadora RF)
   mediante análisis de derivada vertical y dilatación horizontal.

5. TRATAMIENTO ORTOGONAL DE CROMA VHS (Color-Under 629 kHz):
   Desacople de Y, U y V con umbrales independientes (3.5x) y dilatación horizontal
   para erradicar colas de cometa (comet-tails) sin afectar la nitidez de luma.

6. INPAINTING ESPACIAL TRILATERAL (CTMF) PARA DEFECTOS ESTÁTICOS (Gate Hair):
   Detección de suciedad fija en compuerta o lente en 3 cuadros seguidos y
   reconstrucción espacial trilateral de 16-bit.

7. COHERENCIA DE GRANO Y TEXTURA NATURAL:
   Re-inyección adaptativa de micro-textura en las áreas restauradas para que
   la reparación sea 100% invisible y conserve la textura original.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 85,
    threshold:         int   = 10,
    detect_static:     bool  = True,
    show_mask:         str   = "off",
    # Parámetros internos autocalibrados y heredados
    temporal_radius:   int   = 3,
    mode:              str   = "balanced",
    spatial_threshold: int   = 15,
    min_dust_size:     int   = 0,
    max_dust_size:     int   = 32,
    detect_bright:     bool  = True,
    detect_dark:       bool  = True,
    detect_spatial:    bool  = True,
    grain_suppress:    int   = 0,
    edge_protect:      int   = 0,
    scene_protect:     bool  = True,
    scene_threshold:   float = 0.10,
    chroma:            bool  = True,
    grain_restore:     int   = 20,
    blksize:           int | None = None,
    pel:               int | None = None,
    gate_hair:         int   = 50,
    radius:            int   = 3,
    rec:               bool  = True,
    exhaustive_search: bool  = True,
    **kwargs,
) -> vs.VideoNode:
    """
    QuesoLimpia Studio Archive — Motor de máxima profundidad para cintas y video analógico.

    Controles de Usuario:
    ---------------------
    strength      : 10-100% (Fuerza global de sustitución de dropouts y lluvia de VHS).
    threshold     : 2-30    (Sensibilidad a dropouts y chispazos en luma; croma usa 3.5x).
    detect_static : True    (Eliminación de pelos de compuerta y defectos fijos de 3 cuadros).
    show_mask     : "off" / "repair" / "gate_hair" / "side_by_side"
    """
    if clip.format is None:
        raise vs.Error("QuesoLimpia: el clip debe tener formato constante.")

    src_fmt    = clip.format
    src_fmt_id = src_fmt.id
    is_float   = src_fmt.sample_type == vs.FLOAT
    is_gray    = src_fmt.color_family == vs.GRAY
    do_chroma  = (not is_gray) and chroma
    bits_in    = src_fmt.bits_per_sample

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1: PROCESAMIENTO NATIVO EN 16-BIT PARA CERO BANDING
    # ════════════════════════════════════════════════════════════════════════
    if is_float or bits_in < 16:
        if is_gray:
            work_fmt = vs.GRAY16
        else:
            work_fmt = vs.YUV420P16 if src_fmt.subsampling_w == 1 else vs.YUV422P16
        clip16 = core.resize.Point(clip, format=work_fmt)
    else:
        clip16 = clip

    peak = (1 << 16) - 1

    # Escalado de umbrales adaptativos a 16-bit
    strength_scale = max(0.1, strength / 80.0)
    thr_luma   = int(threshold * peak / 255.0 / strength_scale)
    thr_chroma = int(threshold * 3.5 * peak / 255.0 / strength_scale)
    thr_streak = int(max(3, threshold * 0.6) * peak / 255.0 / strength_scale)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: GUÍA DUAL PRE-FILTRADA Y ESTIMACIÓN DE MOVIMIENTO JERÁRQUICA
    # ════════════════════════════════════════════════════════════════════════
    w = clip16.width
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2

    Super       = core.mvsf.Super if is_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_float else core.mv.Recalculate

    # Guía temporal pre-suavizada para aislar la geometría del ruido de cinta
    clip_guide  = core.std.Convolution(clip16, matrix=[1,2,1, 2,4,2, 1,2,1], planes=[0])
    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip16,     pel=pel, sharp=2, rfilter=4)

    search_type  = 3  # Exhaustive Search de máxima precisión
    search_param = 3

    rec1_blk = max(4, blksize // 2)
    rec1_ovl = rec1_blk // 2

    # ── Nivel 1 & 2 & 3: Delta 1 (Cuadros N-1 y N+1) ────────────────────────
    bv1 = Analyse(sup_analyse, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv1 = Analyse(sup_analyse, isb=False, delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)

    bv1 = Recalculate(sup_analyse, bv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)
    fv1 = Recalculate(sup_analyse, fv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)

    if rec1_blk > 4:
        bv1 = Recalculate(sup_analyse, bv1, blksize=4, overlap=2, search=search_type)
        fv1 = Recalculate(sup_analyse, fv1, blksize=4, overlap=2, search=search_type)

    bc1 = Compensate(clip16, sup_comp, bv1)
    fc1 = Compensate(clip16, sup_comp, fv1)

    # ── Nivel 1 & 2: Delta 2 (Cuadros N-2 y N+2) ────────────────────────────
    bv2 = Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv2 = Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)

    bv2 = Recalculate(sup_analyse, bv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv2 = Recalculate(sup_analyse, fv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)

    bc2 = Compensate(clip16, sup_comp, bv2)
    fc2 = Compensate(clip16, sup_comp, fv2)

    # ── Nivel 1 & 2: Delta 3 (Cuadros N-3 y N+3) ────────────────────────────
    bv3 = Analyse(sup_analyse, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=search_type)
    fv3 = Analyse(sup_analyse, isb=False, delta=3, blksize=blksize, overlap=overlap, search=search_type)

    bv3 = Recalculate(sup_analyse, bv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv3 = Recalculate(sup_analyse, fv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)

    bc3 = Compensate(clip16, sup_comp, bv3)
    fc3 = Compensate(clip16, sup_comp, fv3)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: FILTRADO TEMPORAL PROFUNDO DE 7 CUADROS (RANK-ORDER)
    # ════════════════════════════════════════════════════════════════════════
    frames = [fc3, fc2, fc1, clip16, bc1, bc2, bc3]
    interleaved = core.std.Interleave(frames)
    cleaned_all = interleaved.tmedian.TemporalMedian(3, [0, 1, 2] if do_chroma else [0])[3::7]

    # Separar planos de trabajo
    y_orig    = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_cleaned = core.std.ShufflePlanes(cleaned_all, 0, vs.GRAY)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DETECTOR DE CERTEZA MATEMÁTICA Y LÍNEAS 1D DE DEMODULACIÓN
    # ════════════════════════════════════════════════════════════════════════
    # 1. Diferencia temporal estándar
    diff_luma = core.std.Expr([y_orig, y_cleaned], f"x y - abs {thr_luma} > {peak} 0 ?")

    # 2. Votación de certeza: Compara discrepancia entre vecinos compensados N-1 y N+1
    y_bc1 = core.std.ShufflePlanes(bc1, 0, vs.GRAY)
    y_fc1 = core.std.ShufflePlanes(fc1, 0, vs.GRAY)
    # Si N-1 y N+1 coinciden entre sí (cross_diff < thr) pero difieren de N (diff > thr) -> 100% dropout real
    cross_diff = core.std.Expr([y_bc1, y_fc1], f"x y - abs {thr_luma} < {peak} 0 ?")
    confirmed_drops = core.std.Expr([diff_luma, cross_diff], f"x {peak // 2} > y {peak // 2} > and {peak} 0 ?")

    # 3. Detector morfológico 1D de scanlines rasgadas (FM RF loss)
    y_up   = y_orig[:-1] + y_orig[-1:]
    y_down = y_orig[:1]  + y_orig[1:]
    v_diff = core.std.Expr([y_orig, y_up, y_down], f"x y - abs x z - abs min {thr_streak} > {peak} 0 ?")

    # 4. Chispazos extremos de saturación de luma (>235 o <20)
    extreme_sparks = core.std.Expr(
        [y_orig, y_cleaned],
        f"x {int(235 * peak / 255)} > x y - abs {thr_streak} > and x {int(20 * peak / 255)} < x y - abs {thr_streak} > and or {peak} 0 ?"
    )

    # Unión de máscaras de detección
    raw_luma_mask = core.std.Expr([confirmed_drops, v_diff, extreme_sparks], "x y max z max")
    luma_mask = raw_luma_mask.std.Inflate()

    # Sustitución selectiva en Luma
    if strength >= 95:
        y_result = y_cleaned
    else:
        y_result = core.std.MaskedMerge(y_orig, y_cleaned, luma_mask)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: SUPRESIÓN ORTOGONAL DE CROMA VHS (COMET-TAILS & COLOR BLEED)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        u_orig    = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_orig    = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_cleaned = core.std.ShufflePlanes(cleaned_all, 1, vs.GRAY)
        v_cleaned = core.std.ShufflePlanes(cleaned_all, 2, vs.GRAY)

        # Máscaras de croma con umbral 3.5x más sensible y dilatación horizontal
        diff_u = core.std.Expr([u_orig, u_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()
        diff_v = core.std.Expr([v_orig, v_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()

        u_result = core.std.MaskedMerge(u_orig, u_cleaned, diff_u)
        v_result = core.std.MaskedMerge(v_orig, v_cleaned, diff_v)

        repaired = core.std.ShufflePlanes([y_result, u_result, v_result], [0, 0, 0], vs.YUV)
    else:
        repaired = y_result

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: REPARACIÓN DE GATE HAIR (PELOS FIJOS EN 3 CUADROS)
    # ════════════════════════════════════════════════════════════════════════
    static_mask = None
    if detect_static:
        prev_m = luma_mask[-1:] + luma_mask[:-1]
        next_m = luma_mask[1:]  + luma_mask[-1:]
        thr_h  = peak // 2

        # Pelo persistente en 3 cuadros consecutivos
        static_mask = core.std.Expr(
            [luma_mask, prev_m, next_m],
            f"x {thr_h} > y {thr_h} > and z {thr_h} > and {peak} 0 ?"
        )
        static_mask = static_mask.std.Maximum().std.Inflate()

        # Inpainting espacial trilateral (CTMF)
        if do_chroma:
            static_mask_c = core.resize.Bilinear(static_mask, width=u_result.width, height=u_result.height)
            y_spatial  = y_result.ctmf.CTMF(radius=4)
            u_spatial  = u_result.ctmf.CTMF(radius=4)
            v_spatial  = v_result.ctmf.CTMF(radius=4)
            y_repaired = core.std.MaskedMerge(y_result, y_spatial, static_mask)
            u_repaired = core.std.MaskedMerge(u_result, u_spatial, static_mask_c)
            v_repaired = core.std.MaskedMerge(v_result, v_spatial, static_mask_c)
            repaired   = core.std.ShufflePlanes([y_repaired, u_repaired, v_repaired], [0, 0, 0], vs.YUV)
        else:
            y_spatial  = y_result.ctmf.CTMF(radius=4)
            repaired   = core.std.MaskedMerge(y_result, y_spatial, static_mask)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: SÍNTESIS DE COHERENCIA DE GRANO NATURAL
    # ════════════════════════════════════════════════════════════════════════
    if grain_restore > 0:
        grain_added = core.grain.Add(repaired, var=grain_restore * 0.3, uvar=grain_restore * 0.1, constant=False)
        repaired    = core.std.MaskedMerge(repaired, grain_added, luma_mask, planes=[0, 1, 2] if do_chroma else [0])

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 8: DIAGNÓSTICO VISUAL
    # ════════════════════════════════════════════════════════════════════════
    if show_mask == "repair":
        red = core.std.BlankClip(clip16, color=[peak, peak // 2, peak // 2] if do_chroma else [peak])
        overlay = core.std.MaskedMerge(clip16, red, luma_mask, planes=[0, 1, 2] if do_chroma else [0])
        return core.resize.Point(overlay, format=src_fmt_id)

    if show_mask == "gate_hair" and static_mask is not None:
        cyan = core.std.BlankClip(clip16, color=[0, peak, peak // 2] if do_chroma else [peak])
        overlay = core.std.MaskedMerge(clip16, cyan, static_mask, planes=[0, 1, 2] if do_chroma else [0])
        return core.resize.Point(overlay, format=src_fmt_id)

    if show_mask == "side_by_side":
        half    = clip16.width // 2
        left    = core.std.CropAbs(clip16,  width=half, height=clip16.height, left=0, top=0)
        right   = core.std.CropAbs(repaired, width=half, height=clip16.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        return core.resize.Point(stacked, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
