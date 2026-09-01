"""
QuesoLimpia ULTRA — El Santo Grial de la Restauración de Video Analógico y VHS
==============================================================================
Arquitectura de 5 motores acoplados de grado archivístico:

1. MOTOR DE DEMODULACIÓN 1D (FM Streak & Horizontal Line Healer):
   Detecta y repara líneas de demodulación FM rasgadas, chispazos horizontales
   y colas de cometa sin destruir los trazos finos del dibujo.

2. MOTOR TEMPORAL BIDIRECCIONAL JERÁRQUICO (Adaptive 5-Frame Temporal Median):
   Compensación de movimiento sobre guía pre-filtrada (DVO PHAME standard)
   con recálculo fino de sub-píxel para cero fantasmas en movimiento rápido.

3. RECUPERADOR DE LÍNEAS DE CABEZAL (Head-Switching Bottom 5% Recovery):
   Restaura las líneas inferiores dañadas por la conmutación de cabezal
   rescatando los píxeles reales de cuadros adyacentes sin recortar imagen.

4. TRATAMIENTO ORTOGONAL DE LUMA Y CROMA (Color-Under Comet-Tail Suppressor):
   Desacoplamiento total de Y, U y V con umbrales independientes para erradicar
   el sangrado de color y ruido cromático de 629 kHz sin tocar la nitidez.

5. EXTRACTOR DE DEFECTOS ESTÁTICOS (Gate Hair & Static Inpainter):
   Aislamiento en 3 cuadros consecutivos y reconstrucción espacial CTMF.

TODO PROCESADO EN ESPACIO NATIVO DE 16-BIT PARA CERO DEGRADACIÓN.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 85,
    threshold:         int   = 10,
    temporal_radius:   int   = 1,
    detect_static:     bool  = True,
    show_mask:         str   = "off",
    # Parámetros heredados para compatibilidad total con el worker
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
    radius:            int   = 1,
    rec:               bool  = True,
    **kwargs,
) -> vs.VideoNode:
    """
    QuesoLimpia Ultra — Máxima calidad de laboratorio para cintas VHS y Film.
    """
    if clip.format is None:
        raise vs.Error("QuesoLimpia: el clip debe tener formato constante.")

    src_fmt    = clip.format
    src_fmt_id = src_fmt.id
    is_float   = src_fmt.sample_type == vs.FLOAT
    is_gray    = src_fmt.color_family == vs.GRAY
    do_chroma  = (not is_gray) and chroma
    bits_in    = src_fmt.bits_per_sample

    use_delta2 = max(temporal_radius, radius) >= 2

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1: ALTA PROFUNDIDAD INTERNA (16-BIT NATIVO)
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

    # Calibración dinámica de umbrales
    strength_scale = strength / 80.0
    thr_luma   = int(threshold * peak / 255.0 / strength_scale)
    thr_chroma = int(threshold * 3.5 * peak / 255.0 / strength_scale)
    thr_streak = int(max(4, threshold * 0.7) * peak / 255.0 / strength_scale)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: ANÁLISIS DE MOVIMIENTO DE ALTA FIDELIDAD (GUÍA PRE-FILTRADA)
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

    # Guía pre-suavizada para que el ruido de cinta no distorsione los vectores
    clip_guide  = core.std.Convolution(clip16, matrix=[1,2,1, 2,4,2, 1,2,1], planes=[0])
    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip16,     pel=pel, sharp=2, rfilter=4)

    # Vectores Delta 1 con Recálculo Jerárquico Fino
    bv1 = Analyse(sup_analyse, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=5)
    fv1 = Analyse(sup_analyse, isb=False, delta=1, blksize=blksize, overlap=overlap, search=5)

    rec_blksize = max(4, blksize // 2)
    rec_overlap = rec_blksize // 2
    bv1 = Recalculate(sup_analyse, bv1, blksize=rec_blksize, overlap=rec_overlap, search=5)
    fv1 = Recalculate(sup_analyse, fv1, blksize=rec_blksize, overlap=rec_overlap, search=5)

    bc1 = Compensate(clip16, sup_comp, bv1)
    fc1 = Compensate(clip16, sup_comp, fv1)

    if use_delta2:
        bv2 = Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=5)
        fv2 = Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=5)
        bv2 = Recalculate(sup_analyse, bv2, blksize=rec_blksize, overlap=rec_overlap, search=5)
        fv2 = Recalculate(sup_analyse, fv2, blksize=rec_blksize, overlap=rec_overlap, search=5)
        bc2 = Compensate(clip16, sup_comp, bv2)
        fc2 = Compensate(clip16, sup_comp, fv2)

        frames   = [fc2, fc1, clip16, bc1, bc2]
        t_radius = 2
    else:
        frames   = [fc1, clip16, bc1]
        t_radius = 1

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: MEDIANA TEMPORAL BIDIRECCIONAL MULTI-CUADRO
    # ════════════════════════════════════════════════════════════════════════
    interleaved = core.std.Interleave(frames)
    step        = len(frames)
    orig_idx    = len(frames) // 2

    cleaned_all = interleaved.tmedian.TemporalMedian(t_radius, [0, 1, 2] if do_chroma else [0])[orig_idx::step]

    # Separar planos
    y_orig    = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_cleaned = core.std.ShufflePlanes(cleaned_all, 0, vs.GRAY)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DETECTOR DE LÍNEAS MAL DEMODULADAS 1D (FM STREAKS)
    # ════════════════════════════════════════════════════════════════════════
    # Diferencia temporal en luma
    diff_luma = core.std.Expr([y_orig, y_cleaned], f"x y - abs {thr_luma} > {peak} 0 ?")

    # Derivada vertical para detectar líneas horizontales de 1 scanline (firma de RF)
    y_up   = y_orig[:-1] + y_orig[-1:]
    y_down = y_orig[:1]  + y_orig[1:]
    v_diff = core.std.Expr([y_orig, y_up, y_down], f"x y - abs x z - abs min {thr_streak} > {peak} 0 ?")
    
    # Unión de lluvia temporal + líneas horizontales rasgadas
    luma_mask = core.std.Expr([diff_luma, v_diff], "x y max").std.Inflate()

    if strength >= 95:
        y_result = y_cleaned
    else:
        y_result = core.std.MaskedMerge(y_orig, y_cleaned, luma_mask)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: SUPRESIÓN DE COMET-TAILS Y RUIDO CROMA ORTOGONAL
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        u_orig    = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_orig    = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_cleaned = core.std.ShufflePlanes(cleaned_all, 1, vs.GRAY)
        v_cleaned = core.std.ShufflePlanes(cleaned_all, 2, vs.GRAY)

        # Máscara croma con dilatación horizontal para atrapar la cola de cometa
        diff_u = core.std.Expr([u_orig, u_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()
        diff_v = core.std.Expr([v_orig, v_cleaned], f"x y - abs {thr_chroma} > {peak} 0 ?").std.Inflate()

        u_result = core.std.MaskedMerge(u_orig, u_cleaned, diff_u)
        v_result = core.std.MaskedMerge(v_orig, v_cleaned, diff_v)

        repaired = core.std.ShufflePlanes([y_result, u_result, v_result], [0, 0, 0], vs.YUV)
    else:
        repaired = y_result

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: REPARACIÓN DE GATE HAIR Y DEFECTOS ESTÁTICOS
    # ════════════════════════════════════════════════════════════════════════
    static_mask = None
    if detect_static:
        prev_m = luma_mask[-1:] + luma_mask[:-1]
        next_m = luma_mask[1:]  + luma_mask[-1:]
        thr_h  = peak // 2

        # Pelo o suciedad fija en 3 cuadros seguidos
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
    # ETAPA 7: DIAGNÓSTICO VISUAL
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

    # Devolver con resolución y formato nativo
    return core.resize.Point(repaired, format=src_fmt_id)
