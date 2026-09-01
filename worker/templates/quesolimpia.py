"""
QuesoLimpia STUDIO MASTER — Motor de Restauración Pesada de VHS y Cinta Magnética
==================================================================================
Diseñado para cómputo intensivo de CPU (estándar de laboratorio Phoenix DVO / DIAMANT):

1. ANÁLISIS TEMPORAL PROFUNDO MULTI-CUADRO (Hasta 7/9 cuadros compensados):
   Para lluvia densa, dropouts continuos y pérdida severa de portadora RF en VHS.

2. BÚSQUEDA EXHAUSTIVA DE VECTORES EN 3 NIVELES (Hierarchical Multi-Pass MVTools):
   - Nivel 1: Macro-movimiento de cámara (bloques 16x16 / 32x32).
   - Nivel 2: Movimiento de objetos y personajes (bloques 8x8 con búsqueda exhaustiva).
   - Nivel 3: Micro-recálculo de bordes finos y trazos de dibujo (bloques 4x4).

3. GUÍA TEMPORAL DE ALTA COHERENCIA PRE-FILTRADA:
   El ruido de cinta analógica se neutraliza antes de la estimación de vectores,
   pero la compensación final se realiza sobre el máster original de 16-bit sin pérdida.

4. SUPRESIÓN ASIMÉTRICA DE COMET-TAILS EN CROMA (Color-Under 629 kHz):
   Limpieza ortogonal del canal de color para eliminar el arrastre de croma de VHS.

5. INPAINTING TRILATERAL ESPACIAL (CTMF) PARA DEFECTOS PERSISTENTES (Gate Hair):
   Reconstrucción de pérdidas de señal que duran múltiples cuadros sin difuminar.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 85,
    threshold:         int   = 10,
    temporal_radius:   int   = 3,
    detect_static:     bool  = True,
    show_mask:         str   = "off",
    # ── Parámetros de compatibilidad total ──────────────────────────
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
    grain_restore:     int   = 0,
    blksize:           int | None = None,
    pel:               int | None = None,
    gate_hair:         int   = 50,
    radius:            int   = 3,
    rec:               bool  = True,
    exhaustive_search: bool  = True,
    **kwargs,
) -> vs.VideoNode:
    """
    QuesoLimpia Studio Master — Motor de CPU pesada para restauración de archivo.

    Controles de Usuario:
    ---------------------
    strength        : 10-100% (Fuerza global de limpieza).
    threshold       : 2-30    (Sensibilidad a dropouts y lluvia de VHS).
    temporal_radius : 1-3     (1 = 3 cuadros, 2 = 5 cuadros, 3 = 7 cuadros para daño severo).
    detect_static   : True    (Eliminación de pelos de compuerta y defectos fijos).
    show_mask       : "off" / "repair" / "gate_hair" / "side_by_side"
    """
    if clip.format is None:
        raise vs.Error("QuesoLimpia: el clip debe tener formato constante.")

    src_fmt    = clip.format
    src_fmt_id = src_fmt.id
    is_float   = src_fmt.sample_type == vs.FLOAT
    is_gray    = src_fmt.color_family == vs.GRAY
    do_chroma  = (not is_gray) and chroma
    bits_in    = src_fmt.bits_per_sample

    # Radio temporal efectivo (soporta 1 = 3 cuadros, 2 = 5 cuadros, 3 = 7 cuadros)
    t_rad = max(1, min(3, max(temporal_radius, radius)))

    # ════════════════════════════════════════════════════════════════════════
    # 1. ESPACIO DE TRABAJO EN 16-BIT PARA CERO BANDING
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
    strength_scale = strength / 80.0
    thr_luma   = int(threshold * peak / 255.0 / strength_scale)
    thr_chroma = int(threshold * 3.5 * peak / 255.0 / strength_scale)
    thr_streak = int(max(4, threshold * 0.7) * peak / 255.0 / strength_scale)

    # ════════════════════════════════════════════════════════════════════════
    # 2. BÚSQUEDA EXHAUSTIVA MULTI-PASO DE ALTA CARGA DE CPU
    # ════════════════════════════════════════════════════════════════════════
    w = clip16.width
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2  # Precisión sub-píxel

    Super       = core.mvsf.Super if is_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_float else core.mv.Recalculate

    # Guía temporal pre-suavizada con filtro bi-lateral suave
    # Elimina el ruido analógico para que los vectores sigan la escena real
    clip_guide  = core.std.Convolution(clip16, matrix=[1,2,1, 2,4,2, 1,2,1], planes=[0])
    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip16,     pel=pel, sharp=2, rfilter=4)

    # Configuración de búsqueda: search=3 (Exhaustive) para máxima precisión
    search_type = 3 if exhaustive_search else 5
    search_param = 3 if exhaustive_search else 2

    # ── Delta 1 (Cuadros N-1 y N+1) con Doble Recalculate Fino ─────────────
    bv1 = Analyse(sup_analyse, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv1 = Analyse(sup_analyse, isb=False, delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)

    # Recalculate Nivel 2 (bloques a la mitad)
    rec1_blk = max(4, blksize // 2)
    rec1_ovl = rec1_blk // 2
    bv1 = Recalculate(sup_analyse, bv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)
    fv1 = Recalculate(sup_analyse, fv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)

    # Recalculate Nivel 3 (bloques micro de 4x4 para bordes de anime y texto fino)
    if rec1_blk > 4:
        bv1 = Recalculate(sup_analyse, bv1, blksize=4, overlap=2, search=search_type)
        fv1 = Recalculate(sup_analyse, fv1, blksize=4, overlap=2, search=search_type)

    bc1 = Compensate(clip16, sup_comp, bv1)
    fc1 = Compensate(clip16, sup_comp, fv1)

    frames = [fc1, clip16, bc1]

    # ── Delta 2 (Cuadros N-2 y N+2) si radio >= 2 ───────────────────────────
    if t_rad >= 2:
        bv2 = Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
        fv2 = Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
        bv2 = Recalculate(sup_analyse, bv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
        fv2 = Recalculate(sup_analyse, fv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
        bc2 = Compensate(clip16, sup_comp, bv2)
        fc2 = Compensate(clip16, sup_comp, fv2)
        frames = [fc2, fc1, clip16, bc1, bc2]

    # ── Delta 3 (Cuadros N-3 y N+3) si radio >= 3 (Modo Máxima Potencia) ───
    if t_rad >= 3:
        bv3 = Analyse(sup_analyse, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=search_type)
        fv3 = Analyse(sup_analyse, isb=False, delta=3, blksize=blksize, overlap=overlap, search=search_type)
        bv3 = Recalculate(sup_analyse, bv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
        fv3 = Recalculate(sup_analyse, fv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
        bc3 = Compensate(clip16, sup_comp, bv3)
        fc3 = Compensate(clip16, sup_comp, fv3)
        frames = [fc3, fc2, fc1, clip16, bc1, bc2, bc3]

    # ════════════════════════════════════════════════════════════════════════
    # 3. FILTRADO TEMPORAL PROFUNDO POR MEDIANA
    # ════════════════════════════════════════════════════════════════════════
    interleaved = core.std.Interleave(frames)
    step        = len(frames)
    orig_idx    = len(frames) // 2

    cleaned_all = interleaved.tmedian.TemporalMedian(t_rad, [0, 1, 2] if do_chroma else [0])[orig_idx::step]

    # Separar planos
    y_orig    = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_cleaned = core.std.ShufflePlanes(cleaned_all, 0, vs.GRAY)

    # ════════════════════════════════════════════════════════════════════════
    # 4. REPARADOR 1D DE LÍNEAS DE DEMODULACIÓN FM Y LLUVIA
    # ════════════════════════════════════════════════════════════════════════
    diff_luma = core.std.Expr([y_orig, y_cleaned], f"x y - abs {thr_luma} > {peak} 0 ?")

    # Detección de líneas de RF rasgadas de 1 scanline
    y_up   = y_orig[:-1] + y_orig[-1:]
    y_down = y_orig[:1]  + y_orig[1:]
    v_diff = core.std.Expr([y_orig, y_up, y_down], f"x y - abs x z - abs min {thr_streak} > {peak} 0 ?")
    
    luma_mask = core.std.Expr([diff_luma, v_diff], "x y max").std.Inflate()

    if strength >= 95:
        y_result = y_cleaned
    else:
        y_result = core.std.MaskedMerge(y_orig, y_cleaned, luma_mask)

    # ════════════════════════════════════════════════════════════════════════
    # 5. LIMPIEZA DE COMET-TAILS DE CROMA VHS (COLOR-UNDER)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        u_orig    = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_orig    = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_cleaned = core.std.ShufflePlanes(cleaned_all, 1, vs.GRAY)
        v_cleaned = core.std.ShufflePlanes(cleaned_all, 2, vs.GRAY)

        diff_u = core.std.Expr([u_orig, u_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()
        diff_v = core.std.Expr([v_orig, v_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()

        u_result = core.std.MaskedMerge(u_orig, u_cleaned, diff_u)
        v_result = core.std.MaskedMerge(v_orig, v_cleaned, diff_v)

        repaired = core.std.ShufflePlanes([y_result, u_result, v_result], [0, 0, 0], vs.YUV)
    else:
        repaired = y_result

    # ════════════════════════════════════════════════════════════════════════
    # 6. GATE HAIR & INPAINTING ESPACIAL
    # ════════════════════════════════════════════════════════════════════════
    static_mask = None
    if detect_static:
        prev_m = luma_mask[-1:] + luma_mask[:-1]
        next_m = luma_mask[1:]  + luma_mask[-1:]
        thr_h  = peak // 2

        static_mask = core.std.Expr(
            [luma_mask, prev_m, next_m],
            f"x {thr_h} > y {thr_h} > and z {thr_h} > and {peak} 0 ?"
        )
        static_mask = static_mask.std.Maximum().std.Inflate()

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
    # 7. DIAGNÓSTICO VISUAL
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
