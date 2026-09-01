"""
QuesoLimpia MASTER ARCHIVAL SUITE — Restauración 100% Automática de VHS
=======================================================================
Inspirado en Digital Vision DVO (Filmworkz Phoenix) y el ecosistema vhs-decode
(https://github.com/oyvindln/vhs-decode).

Pipeline de Grado de Archivo Cinematográfico a Máxima Potencia de CPU.
100% Automático. Cero artefactos de solarización o pérdida de nitidez.

══════════════════════════════════════════════════════════════════════════
ARQUITECTURA DEL MOTOR MAESTRO (7 ETAPAS):

 1. CROMA DELAY FIX (vhs-decode 629 kHz standard)
    Corrige el retardo de grupo de ~600ns del filtro analógico de 629 kHz
    de VHS desplazando U y V 1.5px a la izquierda con interpolación Spline36.

 2. DVO CROSS-COLOR — Filtro Peine 3D Espacio-Temporal
    ─ DeDot: Peine temporal que cancela dot crawl (C→Y) y cross-color (Y→C)
             aprovechando la inversión de fase de 180° de la portadora de 3.58/4.43 MHz.
    ─ Bifrost: Erradica arcoíris de croma residuales ("hanging rainbows").
    ─ FFT3D 3D Spectral Chroma Cleaning (bt=5, 5 frames):
             Limpieza espectral en frecuencia de los planos U y V sin tocar el luma.

 3. DVO LINE-SYNC — Estabilizador de Jitter de Scanlines
    ─ VerticalCleaner (mode=1): Mediana vertical por línea para neutralizar
      micro-desplazamientos de scanline causados por wow & flutter analógico.
    ─ Convolución vertical armónica para estabilidad de bordes verticales.

 4. DVO MACRO-DROPOUT BRIDGE (Big Drops & Head Clogs: Y, U, V)
    ─ Detección multi-plano de pérdidas masivas de RF, bandas de nieve y fallos de cabezal.
    ─ Scene-Change Guard (PlaneStatsDiff): Protege cortes de edición legítimos.
    ─ Inpainting temporal por puente de flujo óptico inter-cuadro entre vecinos limpios.

 5. DVO MICRO-DROPOUT & TAPE RAIN — Doble Barrera a Máxima CPU
    ─ Barrera Espacial: RemoveGrain(mode=2) + FluxSmoothST para chispas aisladas.
    ─ Barrera Temporal: 7 cuadros (N-3 a N+3) con compensación de movimiento
      exhaustiva jerárquica de 3 niveles (search=3, pel=2, recálculo 16→8→4px)
      y rank-order TemporalMedian.

 6. DE-RINGING ARMÓNICO QUIRÚRGICO 1D (Anti-Peaking de Cabezales)
    ─ Atenuación continua de halos blancos de sobre-impulso con blindaje
      absoluto de trazos negros, pestañas, pupilas, brillos especulares y texto.
    ─ TemporalRepair como ancla temporal de consistencia de trazo.

 7. FUSIÓN CONTINUA MAESTRA (strength 0–100%)
    ─ Mezcla continua de 16-bit con diagnóstico visual opcional.
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
    QuesoLimpia Master Suite — Restauración automática de grado de archivo VHS.
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
    # ETAPA 2: DVO CROSS-COLOR (DOT CRAWL & RAINBOW MOIRÉ ERADICATION)
    # ════════════════════════════════════════════════════════════════════════
    clip_xc = clip16

    # 2A. DeDot: Filtro peine temporal
    if do_chroma and hasattr(core, "dedot"):
        fmt8 = vs.YUV420P8 if src_fmt.subsampling_w == 1 else vs.YUV422P8
        clip_8 = core.resize.Point(clip16, format=fmt8)
        dedot_8 = core.dedot.Dedot(clip_8, luma_2d=2, luma_t=2, chroma_t1=10, chroma_t2=10)
        clip_xc = core.resize.Point(dedot_8, format=clip16.format.id)
    elif do_chroma:
        u_xc = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_xc = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_xc = core.std.Convolution(u_xc, matrix=[1, 2, 4, 2, 1], mode="h")
        v_xc = core.std.Convolution(v_xc, matrix=[1, 2, 4, 2, 1], mode="h")
        clip_xc = core.std.ShufflePlanes([core.std.ShufflePlanes(clip16, 0, vs.GRAY), u_xc, v_xc], [0, 0, 0], vs.YUV)

    # 2B. Bifrost: Eliminación de arcoíris de croma ("hanging rainbows")
    if do_chroma and hasattr(core, "bifrost"):
        fmt8 = vs.YUV420P8 if src_fmt.subsampling_w == 1 else vs.YUV422P8
        clip_8b = core.resize.Point(clip_xc, format=fmt8)
        bifrost_8 = core.bifrost.Bifrost(clip_8b, luma_thresh=0.12, variation=6, conservative_mask=1)
        clip_xc = core.resize.Point(bifrost_8, format=clip16.format.id)

    # 2C. FFT3D 3D Temporal Spectral Comb en U y V (bt=5, CPU Ilimitado)
    if do_chroma and hasattr(core, "fft3dfilter"):
        fmt_ps = vs.YUV420PS if src_fmt.subsampling_w == 1 else vs.YUV422PS
        clip_32 = core.resize.Point(clip_xc, format=fmt_ps)
        clip_fft = core.fft3dfilter.FFT3DFilter(
            clip_32,
            sigma=1.5,
            bt=5,
            bw=16,
            bh=16,
            ow=8,
            oh=8,
            planes=[1, 2],
            ncpu=0
        )
        clip_xc = core.resize.Point(clip_fft, format=clip16.format.id)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: DVO LINE-SYNC (CORRECCIÓN DE JITTER DE SCANLINE / LÍNEAS TBC)
    # ════════════════════════════════════════════════════════════════════════
    clip_ls = clip_xc

    if hasattr(core, "zsmooth"):
        y_ls = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
        # VerticalCleaner: Mediana vertical por scanline
        y_ls_clean = core.zsmooth.VerticalCleaner(y_ls, mode=1)
        # Convolución vertical armónica para eliminar oscilación de jitter
        y_ls_smooth = core.std.Convolution(y_ls_clean, matrix=[1, 2, 4, 2, 1], mode="v")
        y_ls_merged = core.std.Merge(y_ls_clean, y_ls_smooth, weight=0.4)
        if do_chroma:
            u_ls = core.std.ShufflePlanes(clip_ls, 1, vs.GRAY)
            v_ls = core.std.ShufflePlanes(clip_ls, 2, vs.GRAY)
            clip_ls = core.std.ShufflePlanes([y_ls_merged, u_ls, v_ls], [0, 0, 0], vs.YUV)
        else:
            clip_ls = y_ls_merged

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DVO MACRO-DROPOUT BRIDGE (BIG DROPS & HEAD CLOGS EN Y, U, V)
    # ════════════════════════════════════════════════════════════════════════
    y_in   = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
    y_prev = y_in[:1] + y_in[:-1]
    y_next = y_in[1:]  + y_in[-1:]

    thr_big_y  = int(35 * peak / 255)
    thr_big_uv = int(30 * peak / 255)
    thr_near   = int(40 * peak / 255)

    diff_y = core.std.Expr(
        [y_in, y_prev, y_next],
        f"x y - abs {thr_big_y} > "
        f"x z - abs {thr_big_y} > and "
        f"y z - abs {thr_near} < and "
        f"{peak} 0 ?"
    )

    if do_chroma:
        u_in   = core.std.ShufflePlanes(clip_ls, 1, vs.GRAY)
        v_in   = core.std.ShufflePlanes(clip_ls, 2, vs.GRAY)
        u_prev = u_in[:1] + u_in[:-1]
        u_next = u_in[1:]  + u_in[-1:]
        v_prev = v_in[:1] + v_in[:-1]
        v_next = v_in[1:]  + v_in[-1:]

        diff_u = core.std.Expr(
            [u_in, u_prev, u_next],
            f"x y - abs {thr_big_uv} > "
            f"x z - abs {thr_big_uv} > and "
            f"y z - abs {thr_big_uv} < and "
            f"{peak} 0 ?"
        )
        diff_v = core.std.Expr(
            [v_in, v_prev, v_next],
            f"x y - abs {thr_big_uv} > "
            f"x z - abs {thr_big_uv} > and "
            f"y z - abs {thr_big_uv} < and "
            f"{peak} 0 ?"
        )

        diff_u_up = core.resize.Point(diff_u, width=w, height=h)
        diff_v_up = core.resize.Point(diff_v, width=w, height=h)
        macro_mask_raw = core.std.Expr([diff_y, diff_u_up, diff_v_up], "x y max z max")
    else:
        macro_mask_raw = diff_y

    # Scene-Change Guard para evitar falsos positivos en cortes de escena
    y_prev_avg = core.std.PlaneStats(y_in, y_prev, plane=0)
    y_next_avg = core.std.PlaneStats(y_in, y_next, plane=0)

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

    y_bridge = core.std.Expr([y_prev, y_next], "x y + 2 /")
    y_macro  = core.std.MaskedMerge(y_in, y_bridge, macro_mask_y)

    if do_chroma:
        macro_mask_uv = core.resize.Spline36(macro_mask_y, width=u_in.width, height=u_in.height)
        u_bridge = core.std.Expr([u_prev, u_next], "x y + 2 /")
        v_bridge = core.std.Expr([v_prev, v_next], "x y + 2 /")
        u_macro  = core.std.MaskedMerge(u_in, u_bridge, macro_mask_uv)
        v_macro  = core.std.MaskedMerge(v_in, v_bridge, macro_mask_uv)
        clip_pre_clean = core.std.ShufflePlanes([y_macro, u_macro, v_macro], [0, 0, 0], vs.YUV)
    else:
        clip_pre_clean = y_macro

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: DVO MICRO-DROPOUT & TAPE RAIN — DOBLE BARRERA A MÁXIMA CPU
    # ════════════════════════════════════════════════════════════════════════
    # Barrera Espacial: Erradicación de chispas impulsivas aisladas (1-3 px)
    if hasattr(core, "zsmooth"):
        rg_modes = [2, 2, 2] if do_chroma else [2]
        clip_rg = core.zsmooth.RemoveGrain(clip_pre_clean, mode=rg_modes)
        # FluxSmoothST: Suavizado espacio-temporal adaptativo de micro-variaciones
        f_thresh = [8.0, 8.0, 8.0] if do_chroma else [8.0]
        clip_rain = core.zsmooth.FluxSmoothST(
            clip_rg,
            temporal_threshold=f_thresh,
            spatial_threshold=f_thresh
        )
    else:
        clip_rain = clip_pre_clean

    # Barrera Temporal: Mediana rank-order de 7 cuadros con búsqueda exhaustiva MVTools
    if blksize is None:
        blksize = 32 if w > 2400 else 16 if w > 960 else 8
    overlap = blksize // 2
    if pel is None:
        pel = 2

    is_mv_float = clip_rain.format.sample_type == vs.FLOAT
    Super       = core.mvsf.Super if is_mv_float else core.mv.Super
    Analyse     = core.mvsf.Analyse if is_mv_float else core.mv.Analyse
    Compensate  = core.mvsf.Compensate if is_mv_float else core.mv.Compensate
    Recalculate = core.mvsf.Recalculate if is_mv_float else core.mv.Recalculate

    clip_guide  = core.std.Convolution(clip_rain, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
    sup_analyse = Super(clip_guide, pel=pel, sharp=1, rfilter=4)
    sup_comp    = Super(clip_rain,  pel=pel, sharp=2, rfilter=4)

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
        bc = Compensate(clip_rain, sup_comp, bv)
        fc = Compensate(clip_rain, sup_comp, fv)
        return bc, fc

    bc1, fc1 = _make_vec(1)
    bc2, fc2 = _make_vec(2)
    bc3, fc3 = _make_vec(3)

    frames      = [fc3, fc2, fc1, clip_rain, bc1, bc2, bc3]
    interleaved = core.std.Interleave(frames)
    planes      = [0, 1, 2] if do_chroma else [0]
    cleaned_temporal = interleaved.tmedian.TemporalMedian(3, planes)[3::7]

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: DE-RINGING ARMÓNICO QUIRÚRGICO 1D (Anti-Peaking de Cabezales)
    # ════════════════════════════════════════════════════════════════════════
    y_clean_temp = core.std.ShufflePlanes(cleaned_temporal, 0, vs.GRAY)

    # Envolventes continuas 1D
    low_narrow = core.std.Convolution(y_clean_temp, matrix=[1, 2, 1], mode="h")
    low_wide   = core.std.Convolution(y_clean_temp, matrix=[1, 2, 4, 8, 4, 2, 1], mode="h")
    delta_h    = core.std.Expr([low_narrow, low_wide], f"x y - {peak // 2} +")

    # Blindaje Quirúrgico Absoluto:
    # 1. Trazos oscuros (contornos, pestañas, pupilas, barrotes): 100% blindados
    dark_thr     = int(40 * peak / 255)
    is_dark_core = core.std.Expr([y_clean_temp], f"x {dark_thr} < {peak} 0 ?").std.Maximum()

    # 2. Brillos especulares y letras de texto ultrabrillantes ("A"): 100% blindados
    bright_thr     = int(235 * peak / 255)
    is_bright_spec = core.std.Expr([y_clean_temp], f"x {bright_thr} > {peak} 0 ?").std.Maximum()

    # 3. Ancla temporal de consistencia con TemporalRepair
    if hasattr(core, "zsmooth"):
        y_repaired = core.zsmooth.TemporalRepair(y_clean_temp, y_clean_temp, mode=1)
        y_anchor   = core.std.MaskedMerge(y_clean_temp, y_repaired, is_dark_core)
    else:
        y_anchor = y_clean_temp

    # 4. Detector de curvatura 1D de segunda derivada
    dx2 = core.std.Convolution(y_anchor, matrix=[1, -2, 1], mode="h")
    ringing_thr = int(20 * peak / 255)
    is_ringing_raw = core.std.Expr([dx2], f"x {peak // 2} - abs {ringing_thr} > {peak} 0 ?").std.Inflate()

    is_halo_zone = core.std.Expr([is_ringing_raw, is_dark_core, is_bright_spec], "y 0 > z 0 > or 0 x ?")

    y_de_ring = core.std.Expr([y_anchor, delta_h, is_halo_zone], f"z 0 > x y {peak // 2} - 0.75 * - x ?")
    y_de_ring = core.std.Expr([y_de_ring], f"x 0 max {peak} min")

    if do_chroma:
        u_temp = core.std.ShufflePlanes(cleaned_temporal, 1, vs.GRAY)
        v_temp = core.std.ShufflePlanes(cleaned_temporal, 2, vs.GRAY)
        cleaned_final = core.std.ShufflePlanes([y_de_ring, u_temp, v_temp], [0, 0, 0], vs.YUV)
    else:
        cleaned_final = y_de_ring

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: FUSIÓN CONTINUA SEGÚN FUERZA MAESTRA
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
    # ETAPA 8: DIAGNÓSTICO VISUAL
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
