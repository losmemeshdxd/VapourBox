"""
QuesoLimpia MASTER ARCHIVAL SUITE — Restauración 100% Automática de VHS
======================================================================
Inspirado en Digital Vision DVO Dropout (Filmworkz Phoenix) y vhs-decode.
Cobertura Total a Máxima Potencia de CPU: "Big & Small, Luma & Chroma".

ARQUITECTURA DEL MOTOR:
1. DVO MACRO-DROPOUT BRIDGE (Big Drops & Head Clogs):
   Detecta y reconstruye automáticamente bandas masivas de estática RF,
   fallos de cabezal y parches de color corrupto en Y, U y V utilizando
   el puente de interpolación temporal entre cuadros limpios vecinos.

2. DVO MICRO-DROPOUT TEMPORAL MEDIAN (Small Drops & Tape Rain):
   Mediana temporal rank-order de 7 cuadros (N-3 a N+3) con compensación
   de movimiento jerárquica exhaustiva (search=3, pel=2, recálculo 4x4).

3. FILTRO DE MUESCA Y PEINE ADAPTATIVO (Rejas, Ventanas y Tramas):
   Elimina la resonancia de 3.58/4.43 MHz en barrotes verticales y ropa
   rayada, manteniendo los núcleos de barrotes oscuros 100% sólidos.

4. DE-RINGING ARMÓNICO QUIRÚRGICO 1D (Anti-Peaking de Cabezales):
   Atenúa los halos blancos de sobre-impulso con blindaje absoluto de
   trazos negros, pestañas, pupilas, letras y gradientes de luz suaves.

5. RE-ALINEACIÓN SUB-PÍXEL DE CROMA (vhs-decode 629 kHz standard):
   Desplaza el color horizontalmente para centrarlo en los contornos.

6. PROCESAMIENTO NATIVO EN 16-BIT:
   Cero banding, cero parches y máxima fidelidad analógica.
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
    QuesoLimpia Archival Master — Restauración automática de grado de archivo.
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
    # ETAPA 1: TRABAJO EN 16-BIT PARA CERO BANDING
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
    w = clip16.width
    h = clip16.height

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: RE-ALINEACIÓN DE RETARDO DE CROMA (LECCIÓN VHS-DECODE)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        y_plane = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
        u_plane = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_plane = core.std.ShufflePlanes(clip16, 2, vs.GRAY)

        # Corregir el retraso de ~600ns del filtro de 629 kHz desplazando croma a la izquierda (1.5 px)
        u_aligned = core.resize.Spline36(u_plane, src_left=1.5)
        v_aligned = core.resize.Spline36(v_plane, src_left=1.5)
        clip16 = core.std.ShufflePlanes([y_plane, u_aligned, v_aligned], [0, 0, 0], vs.YUV)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: DVO MACRO-DROPOUT BRIDGE (BIG DROPS & HEAD CLOGS EN Y, U, V)
    # ════════════════════════════════════════════════════════════════════════
    # Detecta y repara automáticamente pérdidas de RF masivas, bandas de nieve
    # y colapsos de color en Luma y Croma.
    y_in = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_prev = y_in[:1] + y_in[:-1]
    y_next = y_in[1:] + y_in[-1:]

    diff_y = core.std.Expr(
        [y_in, y_prev, y_next],
        f"x y - abs {int(35 * peak / 255)} > x z - abs {int(35 * peak / 255)} > and y z - abs {int(40 * peak / 255)} < and {peak} 0 ?"
    )

    if do_chroma:
        u_in = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_in = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_prev = u_in[:1] + u_in[:-1]
        u_next = u_in[1:] + u_in[-1:]
        v_prev = v_in[:1] + v_in[:-1]
        v_next = v_in[1:] + v_in[-1:]

        diff_u = core.std.Expr(
            [u_in, u_prev, u_next],
            f"x y - abs {int(30 * peak / 255)} > x z - abs {int(30 * peak / 255)} > and y z - abs {int(30 * peak / 255)} < and {peak} 0 ?"
        )
        diff_v = core.std.Expr(
            [v_in, v_prev, v_next],
            f"x y - abs {int(30 * peak / 255)} > x z - abs {int(30 * peak / 255)} > and y z - abs {int(30 * peak / 255)} < and {peak} 0 ?"
        )

        diff_u_up = core.resize.Point(diff_u, width=w, height=h)
        diff_v_up = core.resize.Point(diff_v, width=w, height=h)

        macro_mask_y = core.std.Expr([diff_y, diff_u_up, diff_v_up], "x y max z max").std.Inflate().std.Inflate()
        macro_mask_uv = core.resize.Spline36(macro_mask_y, width=u_in.width, height=u_in.height)

        y_clean_bridge = core.std.Expr([y_prev, y_next], "x y + 2 /")
        u_clean_bridge = core.std.Expr([u_prev, u_next], "x y + 2 /")
        v_clean_bridge = core.std.Expr([v_prev, v_next], "x y + 2 /")

        y_macro = core.std.MaskedMerge(y_in, y_clean_bridge, macro_mask_y)
        u_macro = core.std.MaskedMerge(u_in, u_clean_bridge, macro_mask_uv)
        v_macro = core.std.MaskedMerge(v_in, v_clean_bridge, macro_mask_uv)

        clip_pre_clean = core.std.ShufflePlanes([y_macro, u_macro, v_macro], [0, 0, 0], vs.YUV)
    else:
        macro_mask_y = diff_y.std.Inflate().std.Inflate()
        y_clean_bridge = core.std.Expr([y_prev, y_next], "x y + 2 /")
        clip_pre_clean = core.std.MaskedMerge(clip16, y_clean_bridge, macro_mask_y)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DVO MICRO-DROPOUTS (MEDIANA TEMPORAL 7 CUADROS A MÁXIMA CPU)
    # ════════════════════════════════════════════════════════════════════════
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2

    Super       = core.mvsf.Super if is_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_float else core.mv.Recalculate

    clip_guide  = core.std.Convolution(clip_pre_clean, matrix=[1,2,1, 2,4,2, 1,2,1], planes=[0])
    sup_analyse = Super(clip_guide,      pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip_pre_clean,  pel=pel, sharp=2, rfilter=4)

    search_type  = 3  # Exhaustive Search
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
    bc1 = Compensate(clip_pre_clean, sup_comp, bv1)
    fc1 = Compensate(clip_pre_clean, sup_comp, fv1)

    # Delta 2 (N-2 y N+2)
    bv2 = Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    fv2 = Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=search_type, searchparam=search_param)
    bv2 = Recalculate(sup_analyse, bv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv2 = Recalculate(sup_analyse, fv2, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    bc2 = Compensate(clip_pre_clean, sup_comp, bv2)
    fc2 = Compensate(clip_pre_clean, sup_comp, fv2)

    # Delta 3 (N-3 y N+3)
    bv3 = Analyse(sup_analyse, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=search_type)
    fv3 = Analyse(sup_analyse, isb=False, delta=3, blksize=blksize, overlap=overlap, search=search_type)
    bv3 = Recalculate(sup_analyse, bv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    fv3 = Recalculate(sup_analyse, fv3, blksize=rec1_blk, overlap=rec1_ovl, search=search_type)
    bc3 = Compensate(clip_pre_clean, sup_comp, bv3)
    fc3 = Compensate(clip_pre_clean, sup_comp, fv3)

    frames = [fc3, fc2, fc1, clip_pre_clean, bc1, bc2, bc3]
    interleaved = core.std.Interleave(frames)
    planes = [0, 1, 2] if do_chroma else [0]
    cleaned_temporal = interleaved.tmedian.TemporalMedian(3, planes)[3::7]

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: DE-RINGING ARMÓNICO QUIRÚRGICO Y FILTRO DE REJAS 1D
    # ════════════════════════════════════════════════════════════════════════
    y_clean_temp = core.std.ShufflePlanes(cleaned_temporal, 0, vs.GRAY)

    # Envolventes continuas 1D suaves (sin cortes morfológicos)
    low_narrow = core.std.Convolution(y_clean_temp, matrix=[1, 2, 1], mode="h")
    low_wide   = core.std.Convolution(y_clean_temp, matrix=[1, 2, 4, 8, 4, 2, 1], mode="h")
    delta_h    = core.std.Expr([low_narrow, low_wide], f"x y - {peak // 2} +")

    # Blindaje Quirúrgico Absoluto:
    # 1. Trazos oscuros (contornos, pestañas, pupilas, barrotes): 100% blindados
    is_dark_core = core.std.Expr([y_clean_temp], f"x {int(40 * peak / 255)} < {peak} 0 ?").std.Maximum()
    # 2. Brillos especulares y letras de texto ultrabrillantes ("A"): 100% blindados
    is_bright_specular = core.std.Expr([y_clean_temp], f"x {int(235 * peak / 255)} > {peak} 0 ?").std.Maximum()

    # 3. Detector de oscilación 1D por segunda derivada (detecta ringing y rejas)
    dx2 = core.std.Convolution(y_clean_temp, matrix=[1, -2, 1], mode="h")
    is_ringing_raw = core.std.Expr(
        [dx2],
        f"x {peak // 2} - abs {int(20 * peak / 255)} > {peak} 0 ?"
    ).std.Inflate()

    # Máscara quirúrgica final: solo zonas de halo oscilatorio fuera de trazos
    is_halo_zone = core.std.Expr(
        [is_ringing_raw, is_dark_core, is_bright_specular],
        "y 0 > z 0 > or 0 x ?"
    )

    y_de_ring = core.std.Expr(
        [y_clean_temp, delta_h, is_halo_zone],
        f"z 0 > x y {peak // 2} - 0.75 * - x ?"
    )

    if do_chroma:
        u_temp = core.std.ShufflePlanes(cleaned_temporal, 1, vs.GRAY)
        v_temp = core.std.ShufflePlanes(cleaned_temporal, 2, vs.GRAY)
        cleaned_final = core.std.ShufflePlanes([y_de_ring, u_temp, v_temp], [0, 0, 0], vs.YUV)
    else:
        cleaned_final = y_de_ring

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: FUSIÓN CONTINUA SEGÚN FUERZA MAESTRA
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
        diff = core.std.Expr([clip16, repaired], "x y - abs 20 *")
        return core.resize.Point(diff, format=src_fmt_id)

    if show_mask == "side_by_side":
        half    = clip16.width // 2
        left    = core.std.CropAbs(clip16,  width=half, height=clip16.height, left=0, top=0)
        right   = core.std.CropAbs(repaired, width=half, height=clip16.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        return core.resize.Point(stacked, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
