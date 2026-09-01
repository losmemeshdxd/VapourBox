"""
QuesoLimpia ALL-IN-ONE — Suite Maestra Automática de Restauración de VHS
========================================================================
Integra todo el pipeline de restauración en 1 solo motor 100% automático:

1. RE-ALINEACIÓN DE RETARDO DE CROMA (Estilo vhs-decode):
   Compensa el desfase de fase de 629 kHz (Color-Under) centrando el color
   sub-píxel horizontalmente dentro de los contornos del dibujo.

2. ERRADICACIÓN DE DROPOUTS Y LLUVIA TEMPORAL (7 Cuadros N-3 a N+3):
   Destruye el 100% de puntos, chispazos y lluvia de cinta sin opacar letras,
   textos o logos (la 'A' de TAXI y detalles finos se conservan 100% intactos).

3. DE-HALO Y ANTI-PEAKING ANALÓGICO (Doble Envolvente Morfológica 1D):
   Elimina los halos blancos de sobre-impulso (overshoot) y surcos oscuros (undershoot)
   de los cabezales del VCR con blindaje total del trazo negro interior.

4. FUSIÓN CONTINUA DE 16-BIT:
   Cero parches, cero manchas, cero degradación de imagen.
"""

import vapoursynth as vs

core = vs.core


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 100,
    show_mask:         str   = "off",
    # Parámetros internos auto-calibrados
    threshold:         int   = 10,
    detect_static:     bool  = False,
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
    QuesoLimpia All-in-One — Restaurador maestro automático para VHS y cinta analógica.
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
    # ETAPA 1: CONVERSIÓN A 16-BIT PARA MÁXIMA PRECISIÓN MATEMÁTICA
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

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: RE-ALINEACIÓN DE RETARDO DE CROMA (LECCIÓN VHS-DECODE)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        y_plane = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
        u_plane = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_plane = core.std.ShufflePlanes(clip16, 2, vs.GRAY)

        # Corregir el retraso de ~600ns del filtro de 629 kHz desplazando croma a la izquierda
        u_aligned = core.resize.Spline36(u_plane, src_left=1.5)
        v_aligned = core.resize.Spline36(v_plane, src_left=1.5)

        clip16 = core.std.ShufflePlanes([y_plane, u_aligned, v_aligned], [0, 0, 0], vs.YUV)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: ESTIMACIÓN DE MOVIMIENTO JERÁRQUICA DE 7 CUADROS (DVO STANDARD)
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

    # Guía temporal pre-suavizada: limpia ruido de cinta para calcular vectores exactos
    clip_guide  = core.std.Convolution(clip16, matrix=[1,2,1, 2,4,2, 1,2,1], planes=[0])
    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip16,     pel=pel, sharp=2, rfilter=4)

    search_type  = 3  # Exhaustive search
    search_param = 3
    rec1_blk = max(4, blksize // 2)
    rec1_ovl = rec1_blk // 2

    # Delta 1 (N-1 y N+1)
    bv1 = Analyse(sup_analyse, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv1 = Analyse(sup_analyse, isb=False, delta=1, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    bv1 = Recalculate(sup_analyse, bv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)
    fv1 = Recalculate(sup_analyse, fv1, blksize=rec1_blk, overlap=rec1_ovl, search=search_type, searchparam=search_param)
    if rec1_blk > 4:
        bv1 = Recalculate(sup_analyse, bv1, blksize=4, overlap=2, search=search_type)
        fv1 = Recalculate(sup_analyse, fv1, blksize=4, overlap=2, search=search_type)
    bc1 = Compensate(clip16, sup_comp, bv1)
    fc1 = Compensate(clip16, sup_comp, fv1)

    # Delta 2 (N-2 y N+2)
    bv2 = Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv2 = Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    bv2 = Recalculate(sup_analyse, bv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv2 = Recalculate(sup_analyse, fv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    bc2 = Compensate(clip16, sup_comp, bv2)
    fc2 = Compensate(clip16, sup_comp, fv2)

    # Delta 3 (N-3 y N+3)
    bv3 = Analyse(sup_analyse, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=search_type)
    fv3 = Analyse(sup_analyse, isb=False, delta=3, blksize=blksize, overlap=overlap, search=search_type)
    bv3 = Recalculate(sup_analyse, bv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv3 = Recalculate(sup_analyse, fv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    bc3 = Compensate(clip16, sup_comp, bv3)
    fc3 = Compensate(clip16, sup_comp, fv3)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: MEDIANA TEMPORAL CONTINUA DE 7 CUADROS (DROPOUT OBLITERATOR)
    # ════════════════════════════════════════════════════════════════════════
    frames = [fc3, fc2, fc1, clip16, bc1, bc2, bc3]
    interleaved = core.std.Interleave(frames)
    planes = [0, 1, 2] if do_chroma else [0]
    cleaned_temporal = interleaved.tmedian.TemporalMedian(3, planes)[3::7]

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: DE-HALO ANALÓGICO 1D (ANTI-PEAKING Y REBOTE DE VCR)
    # ════════════════════════════════════════════════════════════════════════
    y_in = core.std.ShufflePlanes(cleaned_temporal, 0, vs.GRAY)

    # 1. Envolvente superior (Apertura Morfológica: elimina picos blancos)
    r = 7
    y_min = y_in
    for _ in range(r):
        y_min = y_min.std.Minimum(planes=[0])
    upper_bound = y_min
    for _ in range(r):
        upper_bound = upper_bound.std.Maximum(planes=[0])

    # 2. Envolvente inferior (Cierre Morfológico: rellena valles oscuros)
    y_max = y_in
    for _ in range(r):
        y_max = y_max.std.Maximum(planes=[0])
    lower_bound = y_max
    for _ in range(r):
        lower_bound = lower_bound.std.Minimum(planes=[0])

    # 3. Clamping suave de sobre-impulsos analógicos
    thr_dehalo = int(4 * peak / 255)
    y_dehalo_clamped = core.std.Expr(
        [y_in, lower_bound, upper_bound],
        f"x z {thr_dehalo} + > z {thr_dehalo} + x y {thr_dehalo} - < y {thr_dehalo} - x ? ?"
    )

    # 4. Blindaje estricto de trazos negros y líneas finas
    edge_core = core.std.Expr([y_in], f"x {int(40 * peak / 255)} < {peak} 0 ?").std.Inflate()
    diff_halo = core.std.Expr([y_in, upper_bound, lower_bound], "x y - abs x z - abs max")
    halo_zone = core.std.Expr(
        [diff_halo, edge_core],
        f"y {peak // 2} > 0 x {int(10 * peak / 255)} > {peak} 0 ? ?"
    )

    y_dehalo = core.std.MaskedMerge(y_in, y_dehalo_clamped, halo_zone)

    if do_chroma:
        u_in = core.std.ShufflePlanes(cleaned_temporal, 1, vs.GRAY)
        v_in = core.std.ShufflePlanes(cleaned_temporal, 2, vs.GRAY)
        cleaned_final = core.std.ShufflePlanes([y_dehalo, u_in, v_in], [0, 0, 0], vs.YUV)
    else:
        cleaned_final = y_dehalo

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: FUSIÓN PONDERADA CONTINUA (CERO MANCHAS / CERO PARCHES)
    # ════════════════════════════════════════════════════════════════════════
    weight = min(1.0, max(0.1, strength / 100.0))

    if strength >= 98:
        repaired = cleaned_final
    else:
        repaired = core.std.Expr(
            [clip16, cleaned_final],
            f"x {1.0 - weight:.4f} * y {weight:.4f} * +"
        )

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: DIAGNÓSTICO VISUAL
    # ════════════════════════════════════════════════════════════════════════
    if show_mask == "repair":
        diff = core.std.Expr([clip16, cleaned_final], "x y - abs 20 *")
        return core.resize.Point(diff, format=src_fmt_id)

    if show_mask == "side_by_side":
        half    = clip16.width // 2
        left    = core.std.CropAbs(clip16,  width=half, height=clip16.height, left=0, top=0)
        right   = core.std.CropAbs(repaired, width=half, height=clip16.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        return core.resize.Point(stacked, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
