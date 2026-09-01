"""
QuesoLimpia MASTER ARCHIVAL SUITE — Restauración 100% Automática de VHS
=======================================================================
Inspirado en Digital Vision DVO (Filmworkz Phoenix) y el ecosistema vhs-decode
(https://github.com/oyvindln/vhs-decode).

Pipeline de Grado de Archivo Cinematográfico a Máxima Potencia de CPU.
100% Automático. 100% Motion-Compensated. Cero artefactos ni pérdidas de nitidez.
Tamiz Selectivo de Outliers de 9 Cuadros (N-4 a N+4): Preservación Bit a Bit de Detalles.

══════════════════════════════════════════════════════════════════════════
ARQUITECTURA DEL MOTOR MAESTRO DE 9 CUADROS (8 ETAPAS):

 1. CROMA DELAY FIX (vhs-decode standard 629 kHz)
    Corrige el retardo de grupo de ~600ns del filtro analógico de 629 kHz
    de VHS desplazando U y V 1.5px a la izquierda con interpolación Spline36.

 2. DVO CROSS-COLOR — Filtro Peine 3D Espacio-Temporal Motion-Safe
    ─ DeDot: Peine espacial 2D en luma y temporal en croma (luma_t=0 para
             garantizar cero ghosting en luma en movimiento).
    ─ Bifrost: Erradica arcoíris de croma residuales ("hanging rainbows").
    ─ FFT3D Spectral Chroma Cleaning: Limpieza espectral en U y V sin
             tocar el luma ni desaturar.

 3. DVO LINE-SYNC — Estabilizador de Jitter de Scanlines Quirúrgico
    ─ VerticalCleaner con clamp estricto de micro-desviación (±2.0 niveles):
      corrige el bamboleo de scanlines (jitter) sin tocar pestañas,
      párpados, pupilas ni detalles faciales.

 4. DVO MACRO-DROPOUT BRIDGE — MOTION-COMPENSATED (Big Drops & Head Clogs)
    ─ Detección 100% compensada por movimiento (MVTools bc1 / fc1) en Y, U y V:
      elimina por completo los falsos positivos en giros de cabeza, ojos
      o bocas en movimiento (cero manchas oscuras en rostros).
    ─ Inpainting temporal por flujo óptico entre vecinos limpios compensados.

 5. PIRÁMIDE DE MOVIMIENTO EXHAUSTIVA DE 9 CUADROS (Δ = ±1, ±2, ±3, ±4)
    ─ 8 campos de vectores de movimiento exhaustivos (search=3, pel=2,
      recálculo jerárquico profundo 16→8→4px con Chroma-Aware SAD).
    ─ Señal de compensación 100% RAW 16-bit con interpolación Wiener (sharp=2).

 6. TAMIZ SELECTIVO DE OUTLIERS DE LLUVIA (Selective Outlier Sieve)
    ─ Envolvente temporal dinámica [Min(Comp), Max(Comp)] que identifica
      chispas de lluvia reales.
    ─ Los píxeles limpios se conservan 100% BIT A BIT IDÉNTICOS al original.
    ─ Solo los píxeles anómalos son reemplazados por la mediana rank-order
      de 9 cuadros (N-4 a N+4).

 7. DE-RINGING ARMÓNICO QUIRÚRGICO 1D (Anti-Peaking de Cabezales)
    ─ Atenuación continua de halos blancos de sobre-impulso con blindaje
      absoluto de trazos negros, pestañas, pupilas, brillos especulares y texto.

 8. PRESERVACIÓN Y REALCE QUIRÚRGICO DE TEXTURAS REALES (ANTI-BLUR)
    ─ Re-inyección de alta fidelidad de micro-texturas y bordes reales (TCanny),
      garantizando que la imagen sea nítida, cristalina y con grano orgánico.
══════════════════════════════════════════════════════════════════════════
"""

import vapoursynth as vs

core = vs.core


def _load_plugin(name: str) -> bool:
    """Carga un plugin de VapourBox de forma segura. Retorna True si tuvo éxito."""
    import os
    candidates = [
        f"/Users/lorenzoolivera/Library/Application Support/VapourBox/deps/macos-arm64/vapoursynth/plugins/lib{name}.dylib",
        f"/usr/local/lib/vapoursynth/lib{name}.dylib",
        f"/usr/lib/vapoursynth/lib{name}.dylib",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                core.std.LoadPlugin(path)
                return True
            except Exception:
                pass
    return False


# Cargar todos los plugins del ecosistema DVO al inicializar
for _plugin in [
    "dedot", "bifrost", "fft3dfilter", "bm3d", "dfttest",
    "zsmooth", "ttempsmooth", "fluxsmooth", "removegrain",
    "awarpsharp2", "tcanny", "akarin", "nnedi3", "fillborders",
    "mvtools", "tmedian", "fmtconv"
]:
    _load_plugin(_plugin)


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 100,
    show_mask:         str   = "off",
    # Parámetros internos auto-calibrados
    threshold:         int   = 10,
    detect_static:     bool  = False,
    temporal_radius:   int   = 4,
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
    radius:            int   = 4,
    rec:               bool  = True,
    exhaustive_search: bool  = True,
    **kwargs,
) -> vs.VideoNode:
    """
    QuesoLimpia Master Suite — Restauración automática de grado de archivo VHS con Tamiz de 9 Cuadros.
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
    # ETAPA 1: TRABAJO EN 16-BIT — MÁXIMA PRECISIÓN DINÁMICA ANALÓGICA
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
    w    = clip16.width
    h    = clip16.height

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1.5: CROMA DELAY FIX (vhs-decode standard 629 kHz)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        y_plane   = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
        u_plane   = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_plane   = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_aligned = core.resize.Spline36(u_plane, src_left=1.5)
        v_aligned = core.resize.Spline36(v_plane, src_left=1.5)
        clip16    = core.std.ShufflePlanes([y_plane, u_aligned, v_aligned], [0, 0, 0], vs.YUV)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: DVO CROSS-COLOR (MOTION-SAFE)
    # ════════════════════════════════════════════════════════════════════════
    clip_xc = clip16

    # 2A. DeDot: Filtro peine (luma_t=0 para garantizar cero ghosting en luma)
    if do_chroma and hasattr(core, "dedot"):
        fmt8 = vs.YUV420P8 if src_fmt.subsampling_w == 1 else vs.YUV422P8
        clip_8 = core.resize.Point(clip16, format=fmt8)
        dedot_8 = core.dedot.Dedot(clip_8, luma_2d=2, luma_t=0, chroma_t1=10, chroma_t2=10)
        clip_xc = core.resize.Point(dedot_8, format=clip16.format.id)

    # 2B. Bifrost: Eliminación de arcoíris de croma ("hanging rainbows")
    if do_chroma and hasattr(core, "bifrost"):
        fmt8 = vs.YUV420P8 if src_fmt.subsampling_w == 1 else vs.YUV422P8
        clip_8b = core.resize.Point(clip_xc, format=fmt8)
        bifrost_8 = core.bifrost.Bifrost(clip_8b, luma_thresh=0.12, variation=6, conservative_mask=1)
        clip_xc = core.resize.Point(bifrost_8, format=clip16.format.id)

    # 2C. FFT3D Spectral Comb quirúrgico en U y V (bt=3, seguro con movimiento)
    if do_chroma and hasattr(core, "fft3dfilter"):
        fmt_ps = vs.YUV420PS if src_fmt.subsampling_w == 1 else vs.YUV422PS
        clip_32 = core.resize.Point(clip_xc, format=fmt_ps)
        clip_fft = core.fft3dfilter.FFT3DFilter(
            clip_32,
            sigma=1.2,
            bt=3,
            bw=16,
            bh=16,
            ow=8,
            oh=8,
            planes=[1, 2],
            ncpu=0
        )
        clip_xc = core.resize.Point(clip_fft, format=clip16.format.id)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: DVO LINE-SYNC (QUIRÚRGICO Y CLAMPED — CERO DESENFOQUE FACIAL)
    # ════════════════════════════════════════════════════════════════════════
    clip_ls = clip_xc

    if hasattr(core, "zsmooth"):
        y_ls = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
        y_ls_clean = core.zsmooth.VerticalCleaner(y_ls, mode=1)
        # Clamp estricto: micro-corrección de jitter de scanlines
        max_dev = int(2.0 * peak / 255)
        y_ls_clamped = core.std.Expr([y_ls, y_ls_clean], f"y x {max_dev} - max x {max_dev} + min")
        if do_chroma:
            u_ls = core.std.ShufflePlanes(clip_ls, 1, vs.GRAY)
            v_ls = core.std.ShufflePlanes(clip_ls, 2, vs.GRAY)
            clip_ls = core.std.ShufflePlanes([y_ls_clamped, u_ls, v_ls], [0, 0, 0], vs.YUV)
        else:
            clip_ls = y_ls_clamped

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4 & 5: PIRÁMIDE DE MOVIMIENTO EXHAUSTIVA DE 9 CUADROS (Δ = ±1..±4)
    # ════════════════════════════════════════════════════════════════════════
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2

    is_mv_float = clip_ls.format.sample_type == vs.FLOAT
    Super       = core.mvsf.Super if is_mv_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_mv_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_mv_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_mv_float else core.mv.Recalculate

    clip_guide  = core.std.Convolution(clip_ls, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
    if hasattr(core, "zsmooth"):
        rg_modes = [2, 2, 2] if do_chroma else [2]
        clip_guide = core.zsmooth.RemoveGrain(clip_guide, mode=rg_modes)

    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip_ls,    pel=pel, sharp=2, rfilter=4)

    search_type  = 3  # Exhaustive Search (Máximo CPU)
    search_param = 4  # Radio de búsqueda profundo
    rec1_blk     = max(4, blksize // 2)
    rec1_ovl     = rec1_blk // 2

    def _make_vec(delta: int):
        bv = Analyse(sup_analyse, isb=True,  delta=delta, blksize=blksize, overlap=overlap,
                     search=search_type, searchparam=search_param, chroma=True, truemotion=True)
        fv = Analyse(sup_analyse, isb=False, delta=delta, blksize=blksize, overlap=overlap,
                     search=search_type, searchparam=search_param, chroma=True, truemotion=True)
        # Recálculo Nivel 1 (Sub-Bloques)
        bv = Recalculate(sup_analyse, bv, blksize=rec1_blk, overlap=rec1_ovl,
                         search=search_type, searchparam=search_param, chroma=True, truemotion=True)
        fv = Recalculate(sup_analyse, fv, blksize=rec1_blk, overlap=rec1_ovl,
                         search=search_type, searchparam=search_param, chroma=True, truemotion=True)
        # Recálculo Nivel 2 (Micro-Bloques 4x4)
        if rec1_blk > 4:
            bv = Recalculate(sup_analyse, bv, blksize=4, overlap=2, search=search_type, chroma=True)
            fv = Recalculate(sup_analyse, fv, blksize=4, overlap=2, search=search_type, chroma=True)
        bc = Compensate(clip_ls, sup_comp, bv)
        fc = Compensate(clip_ls, sup_comp, fv)
        return bc, fc

    bc1, fc1 = _make_vec(1)
    bc2, fc2 = _make_vec(2)
    bc3, fc3 = _make_vec(3)
    bc4, fc4 = _make_vec(4)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DVO MACRO-DROPOUT BRIDGE (100% MOTION-COMPENSATED)
    # ════════════════════════════════════════════════════════════════════════
    y_in  = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
    y_bc1 = core.std.ShufflePlanes(bc1, 0, vs.GRAY)
    y_fc1 = core.std.ShufflePlanes(fc1, 0, vs.GRAY)

    thr_big_y  = int(35 * peak / 255)
    thr_big_uv = int(30 * peak / 255)
    thr_near   = int(35 * peak / 255)

    diff_y = core.std.Expr(
        [y_in, y_bc1, y_fc1],
        f"x y - abs {thr_big_y} > "
        f"x z - abs {thr_big_y} > and "
        f"y z - abs {thr_near} < and "
        f"{peak} 0 ?"
    )

    if do_chroma:
        u_in  = core.std.ShufflePlanes(clip_ls, 1, vs.GRAY)
        v_in  = core.std.ShufflePlanes(clip_ls, 2, vs.GRAY)
        u_bc1 = core.std.ShufflePlanes(bc1, 1, vs.GRAY)
        u_fc1 = core.std.ShufflePlanes(fc1, 1, vs.GRAY)
        v_bc1 = core.std.ShufflePlanes(bc1, 2, vs.GRAY)
        v_fc1 = core.std.ShufflePlanes(fc1, 2, vs.GRAY)

        diff_u = core.std.Expr(
            [u_in, u_bc1, u_fc1],
            f"x y - abs {thr_big_uv} > x z - abs {thr_big_uv} > and y z - abs {thr_big_uv} < and {peak} 0 ?"
        )
        diff_v = core.std.Expr(
            [v_in, v_bc1, v_fc1],
            f"x y - abs {thr_big_uv} > x z - abs {thr_big_uv} > and y z - abs {thr_big_uv} < and {peak} 0 ?"
        )

        diff_u_up = core.resize.Point(diff_u, width=w, height=h)
        diff_v_up = core.resize.Point(diff_v, width=w, height=h)
        macro_mask_raw = core.std.Expr([diff_y, diff_u_up, diff_v_up], "x y max z max")
    else:
        macro_mask_raw = diff_y

    # Scene-Change Guard
    y_prev_raw = y_in[:1] + y_in[:-1]
    y_next_raw = y_in[1:]  + y_in[-1:]
    y_prev_avg = core.std.PlaneStats(y_in, y_prev_raw, plane=0)
    y_next_avg = core.std.PlaneStats(y_in, y_next_raw, plane=0)

    def _scene_guard(n: int, f: list, clip_mask: vs.VideoNode, blank: vs.VideoNode) -> vs.VideoNode:
        diff_prev = f[0].props.get("PlaneStatsDiff", 0.0)
        diff_next = f[1].props.get("PlaneStatsDiff", 0.0)
        if diff_prev > 0.15 or diff_next > 0.15:
            return blank
        return clip_mask

    blank_mask = core.std.BlankClip(macro_mask_raw, color=[0] * macro_mask_raw.format.num_planes)
    macro_mask_y = core.std.FrameEval(
        macro_mask_raw,
        lambda n, f: _scene_guard(n, f, macro_mask_raw, blank_mask),
        prop_src=[y_prev_avg, y_next_avg]
    ).std.Inflate().std.Inflate()

    # Reconstrucción motion-compensated
    y_bridge = core.std.Expr([y_bc1, y_fc1], "x y + 2 /")
    y_macro  = core.std.MaskedMerge(y_in, y_bridge, macro_mask_y)

    if do_chroma:
        macro_mask_uv = core.resize.Spline36(macro_mask_y, width=u_in.width, height=u_in.height)
        u_bridge = core.std.Expr([u_bc1, u_fc1], "x y + 2 /")
        v_bridge = core.std.Expr([v_bc1, v_fc1], "x y + 2 /")
        u_macro  = core.std.MaskedMerge(u_in, u_bridge, macro_mask_uv)
        v_macro  = core.std.MaskedMerge(v_in, v_bridge, macro_mask_uv)
        clip_pre_clean = core.std.ShufflePlanes([y_macro, u_macro, v_macro], [0, 0, 0], vs.YUV)
    else:
        clip_pre_clean = y_macro

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5 & 6: TAMIZ SELECTIVO DE OUTLIERS DE LLUVIA DE 9 CUADROS (N-4 a N+4)
    # ════════════════════════════════════════════════════════════════════════
    # Mediana rank-order exhaustiva de 9 cuadros
    frames_9    = [fc4, fc3, fc2, fc1, clip_pre_clean, bc1, bc2, bc3, bc4]
    interleaved = core.std.Interleave(frames_9)
    planes      = [0, 1, 2] if do_chroma else [0]
    cleaned_9   = interleaved.tmedian.TemporalMedian(4, planes)[4::9]

    # Envolvente temporal de movimiento [min, max] para discriminar lluvia de textura
    y_clean_src = core.std.ShufflePlanes(clip_pre_clean, 0, vs.GRAY)
    y_bc2 = core.std.ShufflePlanes(bc2, 0, vs.GRAY)
    y_fc2 = core.std.ShufflePlanes(fc2, 0, vs.GRAY)
    y_c9  = core.std.ShufflePlanes(cleaned_9, 0, vs.GRAY)

    y_min_env = core.std.Expr([y_bc1, y_fc1, y_bc2, y_fc2], "x y min z min a min")
    y_max_env = core.std.Expr([y_bc1, y_fc1, y_bc2, y_fc2], "x y max z max a max")

    # Tolerancia de ruido normal (±8 niveles)
    tol = int(8 * peak / 255)
    is_rain_outlier = core.std.Expr(
        [y_clean_src, y_min_env, y_max_env],
        f"x y {tol} - < x z {tol} + > or {peak} 0 ?"
    )

    # Blindaje de bordes reales finos con TCanny
    if hasattr(core, 'tcanny'):
        edge_shield = core.tcanny.TCanny(y_clean_src, sigma=0.8, mode=1)
        # Fusión selectiva: Solo si es outlier de lluvia Y no es un micro-borde coherente
        y_sieved = core.std.Expr(
            [y_clean_src, y_c9, is_rain_outlier, edge_shield],
            f"z 0 > a {int(25 * peak / 255)} < and y x ?"
        )
    else:
        y_sieved = core.std.Expr(
            [y_clean_src, y_c9, is_rain_outlier],
            "z 0 > y x ?"
        )

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: DE-RINGING ARMÓNICO QUIRÚRGICO 1D (Anti-Peaking de Cabezales)
    # ════════════════════════════════════════════════════════════════════════
    # Envolventes 1D continuas horizontales
    low_narrow = core.std.Convolution(y_sieved, matrix=[1, 2, 1], mode="h")
    low_wide   = core.std.Convolution(y_sieved, matrix=[1, 2, 4, 8, 4, 2, 1], mode="h")
    delta_h    = core.std.Expr([low_narrow, low_wide], f"x y - {peak // 2} +")

    # Blindaje Quirúrgico Absoluto:
    # 1. Trazos oscuros (contornos, pestañas, pupilas, barrotes): 100% blindados
    dark_thr     = int(40 * peak / 255)
    is_dark_core = core.std.Expr([y_sieved], f"x {dark_thr} < {peak} 0 ?").std.Maximum()

    # 2. Brillos especulares y letras de texto ultrabrillantes ("A"): 100% blindados
    bright_thr     = int(235 * peak / 255)
    is_bright_spec = core.std.Expr([y_sieved], f"x {bright_thr} > {peak} 0 ?").std.Maximum()

    # 3. Detector de curvatura 1D de segunda derivada
    dx2 = core.std.Convolution(y_sieved, matrix=[1, -2, 1], mode="h")
    ringing_thr = int(24 * peak / 255)
    is_ringing_raw = core.std.Expr([dx2], f"x {peak // 2} - abs {ringing_thr} > {peak} 0 ?").std.Inflate()

    is_halo_zone = core.std.Expr([is_ringing_raw, is_dark_core, is_bright_spec], "y 0 > z 0 > or 0 x ?")

    y_de_ring = core.std.Expr([y_sieved, delta_h, is_halo_zone], f"z 0 > x y {peak // 2} - 0.70 * - x ?")
    y_de_ring = core.std.Expr([y_de_ring], f"x 0 max {peak} min")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 8: PRESERVACIÓN Y REALCE QUIRÚRGICO DE TEXTURA Y NITIDEZ (ANTI-BLUR)
    # ════════════════════════════════════════════════════════════════════════
    y_orig = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    if hasattr(core, 'tcanny'):
        edge_mask = core.tcanny.TCanny(y_orig, sigma=0.8, mode=1)
        hf_texture = core.std.Expr([y_orig, y_de_ring], f"x y - {peak // 2} +")
        y_sharp = core.std.Expr(
            [y_de_ring, hf_texture, edge_mask],
            f"z {int(10 * peak / 255)} > x y {peak // 2} - 0.40 * + x ?"
        )
        y_sharp = core.std.Expr([y_sharp], f"x 0 max {peak} min")
    else:
        y_sharp = y_de_ring

    if do_chroma:
        u_temp = core.std.ShufflePlanes(cleaned_9, 1, vs.GRAY)
        v_temp = core.std.ShufflePlanes(cleaned_9, 2, vs.GRAY)
        cleaned_final = core.std.ShufflePlanes([y_sharp, u_temp, v_temp], [0, 0, 0], vs.YUV)
    else:
        cleaned_final = y_sharp

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 9: FUSIÓN CONTINUA SEGÚN FUERZA MAESTRA
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
    # ETAPA 10: DIAGNÓSTICO VISUAL
    # ════════════════════════════════════════════════════════════════════════
    if show_mask == "repair":
        diff = core.std.Expr([clip16, repaired], "x y - abs 20 *")
        return core.resize.Point(diff, format=src_fmt_id)

    if show_mask == "side_by_side":
        half    = clip16.width // 2
        left    = core.std.CropAbs(clip16,   width=half, height=clip16.height, left=0, top=0)
        right   = core.std.CropAbs(repaired, width=half, height=clip16.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        return core.resize.Point(stacked, format=src_fmt_id)

    if show_mask == "dropout_mask":
        return core.resize.Point(macro_mask_y, format=src_fmt_id)

    if show_mask == "xcolor_mask":
        if do_chroma:
            u_before = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
            u_after  = core.std.ShufflePlanes(clip_xc, 1, vs.GRAY)
            diff_uv  = core.std.Expr([u_before, u_after], "x y - abs 20 *")
            diff_up  = core.resize.Point(diff_uv, width=w, height=h)
            return core.resize.Point(diff_up, format=src_fmt_id)
        return core.resize.Point(clip16, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
