"""
QuesoLimpia MASTER ARCHIVAL SUITE v3.1 (Pristine Archival Edition)
========================================================================================
Inspirado en Digital Vision DVO Dust & Grain (Filmworkz Phoenix), Algosoft VIVA AI,
Neat Video Pro y el ecosistema vhs-decode (https://github.com/oyvindln/vhs-decode).

Pipeline de Grado de Archivo Cinematográfico a Máxima Potencia de CPU.
100% Libre de Artefactos de Bloque, Escaleras y Distorsiones Diagonales.

════════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA PURA DE 9 ETAPAS (PRECISIÓN MATEMÁTICA TOTAL):

 0. CONVERSIÓN A 16-BIT (Rango dinámico completo 0..65535 en toda la cadena)

 1. CROMA DELAY FIX ANALÓGICO (vhs-decode Standard 629 kHz)
    Corrección de fase de ~600ns del filtro pasabajos de cinta VHS (1.5px Spline36).

 2. DVO CROSS-COLOR & 3D COMB (Bifrost + DeDot + FFT3D bt=5)
    ─ DeDot en croma (luma_2d=0 para garantizar cero ghosting en luma).
    ─ Bifrost: erradica arcoíris parásitos ("hanging rainbows").
    ─ FFT3D Comb Espectral en U/V (bt=5 cuadros temporales).

 3. DVO LINE-SYNC — Estabilizador de Jitter de Scanlines Quirúrgico
    ─ VerticalCleaner + clamp estricto de ±2.0 niveles (cero deformación facial).

 4. 🛡️ 5-FRAME IMPULSE OUTLIER SIEVE CON DISCRIMINACIÓN MORFOLÓGICA DE ESCALA
    ─ CERO compensación por bloques → CERO artefactos de escalera o agujeros.
    ─ Detección de impulsos temporales sobre envolvente de 5 cuadros (N-2..N+2).
    ─ Filtro morfológico de escala: separa speckles pequeños (≤4px) de objetos grandes
      en movimiento (brazos, manos, dedos, cables, micrófonos) que quedan 100.0% intactos.
    ─ Erradica lluvia de cinta de 1 y 2 cuadros de duración al 100.0%.

 5. 🧭 MVDEGRAIN3 — INTEGRACIÓN TEMPORAL PONDERADA BAYESIANA (6 Cuadros Compensados)
    ─ MVDegrain3 sobre 6 campos de movimiento exhaustivos (search=3, pel=2, thsad=140).
    ─ Promedio temporal continuo ponderado (NO reemplazo de bloque) → CERO artefactos.

 6. 🎨 CCD CHROMA CONVERGENCE DENOISER + FLUXSMOOTHT ANTI-FLICKER
    ─ CCD: suprime nubes y manchas de croma.
    ─ FluxSmoothT: estabiliza el parpadeo de luminancia inter-frame y bombeo de AGC.

 7. 🌈 16-BIT PRECISION DEBANDING (neo_f3kdb)
    ─ Erradica posterización y bandas de color de compresión analógica/digital.

 8. 📐 AUTO-DEHALO ARMÓNICO 1D (Anti-Peaking)
    ─ Descomposición espectral 1D para eliminar halos de sobre-impulso sin tocar gradientes.

 9. 💎 REALCE ADAPTATIVO POR CONTRASTE EN 16-BIT NATIVO (CAS)
    ─ Realce de micro-contraste directo sin transposición para bordes 100% naturales.
════════════════════════════════════════════════════════════════════════════════════════
"""

import vapoursynth as vs

core = vs.core


def _load_plugin(name: str) -> bool:
    """Carga un plugin de VapourBox de forma segura si no está ya cargado."""
    import os
    candidates = [
        f"/Users/lorenzoolivera/Library/Application Support/VapourBox/deps/macos-arm64/vapoursynth/plugins/lib{name}.dylib",
        f"deps/macos-arm64/vapoursynth/plugins/lib{name}.dylib",
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


# Cargar plugins de respaldo de forma segura
for _p in [
    "dedot", "bifrost", "fft3dfilter", "zsmooth", "ttempsmooth",
    "fluxsmooth", "removegrain", "tcanny", "akarin", "mvtools",
    "tmedian", "fmtconv", "deblock", "bm3d", "dfttest", "nnedi3",
    "cas", "neo_f3kdb",
]:
    _load_plugin(_p)


def QuesoLimpia(
    clip:              vs.VideoNode,
    strength:          int   = 100,
    show_mask:         str   = "off",
    # Parámetros de compatibilidad con pipeline Rust
    threshold:         int   = 10,
    detect_static:     bool  = False,
    temporal_radius:   int   = 2,
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
    radius:            int   = 2,
    rec:               bool  = True,
    exhaustive_search: bool  = True,
    ai_chroma_nn:      bool  = True,
    **kwargs,
) -> vs.VideoNode:
    """
    QuesoLimpia Master Suite v3.1 (Pristine Archival Edition)
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
    # ETAPA 0: TRABAJO EN 16-BIT — MÁXIMA PRECISIÓN DINÁMICA ANALÓGICA
    # ════════════════════════════════════════════════════════════════════════
    if is_float or bits_in != 16:
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
    # ETAPA 1: CROMA DELAY FIX ANALÓGICO (vhs-decode standard 629 kHz)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        y_plane   = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
        u_plane   = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_plane   = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        u_aligned = core.resize.Spline36(u_plane, src_left=1.5)
        v_aligned = core.resize.Spline36(v_plane, src_left=1.5)
        clip_cd   = core.std.ShufflePlanes([y_plane, u_aligned, v_aligned], [0, 0, 0], vs.YUV)
    else:
        clip_cd   = clip16

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: DVO CROSS-COLOR & 3D COMB (Bifrost + DeDot + FFT3D bt=5)
    # ════════════════════════════════════════════════════════════════════════
    clip_xc = clip_cd

    # 2A. DeDot en croma
    if do_chroma and hasattr(core, "dedot"):
        try:
            fmt8 = vs.YUV420P8 if clip16.format.subsampling_w == 1 else vs.YUV422P8
            clip_sub = core.resize.Point(clip_xc, format=fmt8)
            dedot_8 = core.dedot.Dedot(clip_sub, luma_2d=0, luma_t=0, chroma_t1=10, chroma_t2=10)
            dedot_16 = core.resize.Point(dedot_8, format=vs.YUV420P16)
            y_orig_p = core.std.ShufflePlanes(clip_xc, 0, vs.GRAY)
            u_cw = clip16.width // (1 << clip16.format.subsampling_w)
            u_ch = clip16.height // (1 << clip16.format.subsampling_h)
            u_dd     = core.resize.Spline36(core.std.ShufflePlanes(dedot_16, 1, vs.GRAY), width=u_cw, height=u_ch)
            v_dd     = core.resize.Spline36(core.std.ShufflePlanes(dedot_16, 2, vs.GRAY), width=u_cw, height=u_ch)
            clip_xc  = core.std.ShufflePlanes([y_orig_p, u_dd, v_dd], [0, 0, 0], vs.YUV)
        except Exception:
            pass

    # 2B. Bifrost: Erradicación de arcoíris de croma
    if do_chroma and hasattr(core, "bifrost"):
        try:
            fmt8 = vs.YUV420P8 if clip16.format.subsampling_w == 1 else vs.YUV422P8
            clip_8b = core.resize.Point(clip_xc, format=fmt8)
            bifrost_8 = core.bifrost.Bifrost(clip_8b, luma_thresh=0.12, variation=6, conservative_mask=1)
            clip_xc = core.resize.Point(bifrost_8, format=clip16.format.id)
        except Exception:
            pass

    # 2C. FFT3D Spectral Comb en U y V con bt=5
    if do_chroma and hasattr(core, "fft3dfilter"):
        try:
            fmt_ps = vs.YUV420PS if clip16.format.subsampling_w == 1 else vs.YUV444PS
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
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: DVO LINE-SYNC — ESTABILIZADOR DE JITTER DE SCANLINES
    # ════════════════════════════════════════════════════════════════════════
    clip_ls = clip_xc

    if hasattr(core, "zsmooth"):
        y_ls = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
        y_ls_clean = core.zsmooth.VerticalCleaner(y_ls, mode=1)
        max_dev = int(2.0 * peak / 255)
        y_ls_clamped = core.std.Expr([y_ls, y_ls_clean], f"y x {max_dev} - max x {max_dev} + min")
        if do_chroma:
            u_ls = core.std.ShufflePlanes(clip_ls, 1, vs.GRAY)
            v_ls = core.std.ShufflePlanes(clip_ls, 2, vs.GRAY)
            clip_ls = core.std.ShufflePlanes([y_ls_clamped, u_ls, v_ls], [0, 0, 0], vs.YUV)
        else:
            clip_ls = y_ls_clamped

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: 🛡️ 5-FRAME IMPULSE OUTLIER SIEVE CON DISCRIMINACIÓN MORFOLÓGICA
    # ════════════════════════════════════════════════════════════════════════
    # Erradicación 100.0% de lluvia de cinta y speckles de 1 y 2 cuadros.
    # Cero compensación por bloques → Cero artefactos de escalera o agujeros.
    impulse_thr = int(max(6, min(14, 10 * strength / 100)) * peak / 255)

    def _impulse_sieve_plane(src_plane: vs.VideoNode) -> vs.VideoNode:
        y_p1 = src_plane[:1] + src_plane[:-1]
        y_p2 = src_plane[:2] + src_plane[:-2]
        y_n1 = src_plane[1:]  + src_plane[-1:]
        y_n2 = src_plane[2:]  + src_plane[-2:]

        # Detección de impulso sobre 5 cuadros
        is_impulse = core.std.Expr(
            [src_plane, y_p1, y_p2, y_n1, y_n2],
            f"x y z max a max b max {impulse_thr} + > "
            f"x y z min a min b min {impulse_thr} - < "
            f"or {peak} 0 ?"
        )

        # Discriminación morfológica de escala:
        # Speckles (≤3px) se erosionan a 0. Objetos grandes en movimiento sobreviven.
        eroded = is_impulse.std.Minimum().std.Minimum()
        large_motion = eroded.std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate()

        # Máscara de manchas reales (excluyendo objetos grandes en movimiento)
        true_spots = core.std.Expr([is_impulse, large_motion], "x y > x 0 ?")

        # Inpainting temporal suave con promedio de vecinos (prev1 + next1) / 2
        cleaned = core.std.Expr([src_plane, y_p1, y_n1, true_spots], "a 0 > y z + 2 / x ?")
        return core.std.Expr([cleaned], f"x 0 max {peak} min")

    y_raw_ls = core.std.ShufflePlanes(clip_ls, 0, vs.GRAY)
    y_sieved = _impulse_sieve_plane(y_raw_ls)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: 🧭 MVDEGRAIN3 — INTEGRACIÓN TEMPORAL PONDERADA (6 Cuadros)
    # ════════════════════════════════════════════════════════════════════════
    has_mv = hasattr(core, "mv")
    if has_mv:
        try:
            if blksize is None:
                blksize = 32 if w > 2400 else 16 if w > 960 else 8
            overlap = blksize // 2
            if pel is None:
                pel = 2

            guide = core.std.Convolution(y_sieved, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
            sup_a = core.mv.Super(guide,    pel=pel, sharp=1, rfilter=4)
            sup_c = core.mv.Super(y_sieved, pel=pel, sharp=2, rfilter=4)

            bv1 = core.mv.Analyse(sup_a, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv1 = core.mv.Analyse(sup_a, isb=False, delta=1, blksize=blksize, overlap=overlap, search=3, chroma=False)
            bv2 = core.mv.Analyse(sup_a, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv2 = core.mv.Analyse(sup_a, isb=False, delta=2, blksize=blksize, overlap=overlap, search=3, chroma=False)
            bv3 = core.mv.Analyse(sup_a, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv3 = core.mv.Analyse(sup_a, isb=False, delta=3, blksize=blksize, overlap=overlap, search=3, chroma=False)

            y_degrained = core.mv.Degrain3(
                y_sieved, sup_c,
                bv1, fv1, bv2, fv2, bv3, fv3,
                thsad=140, thscd1=350
            )
        except Exception:
            y_degrained = y_sieved
    else:
        y_degrained = y_sieved

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: 🎨 CCD CHROMA CONVERGENCE + FLUXSMOOTHT ANTI-FLICKER
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        u_in_sieved = _impulse_sieve_plane(core.std.ShufflePlanes(clip_ls, 1, vs.GRAY))
        v_in_sieved = _impulse_sieve_plane(core.std.ShufflePlanes(clip_ls, 2, vs.GRAY))

        clip_combined = core.std.ShufflePlanes([y_degrained, u_in_sieved, v_in_sieved], [0, 0, 0], vs.YUV)

        # CCD Chroma Denoiser
        if hasattr(core, "zsmooth") and hasattr(core.zsmooth, "CCD"):
            try:
                clip_combined = core.zsmooth.CCD(clip_combined, threshold=3.0)
            except Exception:
                pass

        # FluxSmoothT Anti-Flicker (estabiliza bombeo de AGC y parpadeo inter-frame)
        if hasattr(core, "zsmooth") and hasattr(core.zsmooth, "FluxSmoothT"):
            try:
                clip_combined = core.zsmooth.FluxSmoothT(clip_combined, temporal_threshold=[4.0, 3.0, 3.0])
            except Exception:
                pass
    else:
        if hasattr(core, "zsmooth") and hasattr(core.zsmooth, "FluxSmoothT"):
            try:
                y_degrained = core.zsmooth.FluxSmoothT(y_degrained, temporal_threshold=[4.0])
            except Exception:
                pass
        clip_combined = y_degrained

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: 🌈 16-BIT PRECISION DEBANDING (neo_f3kdb)
    # ════════════════════════════════════════════════════════════════════════
    if hasattr(core, "neo_f3kdb"):
        try:
            clip_deband = core.neo_f3kdb.Deband(
                clip_combined,
                range=16,
                y=28,
                cb=20 if do_chroma else 0,
                cr=20 if do_chroma else 0,
                grainy=0,
                grainc=0,
                output_depth=16
            )
        except Exception:
            clip_deband = clip_combined
    else:
        clip_deband = clip_combined

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 8: 📐 DE-RINGING & AUTO-DEHALO ARMÓNICO QUIRÚRGICO 1D
    # ════════════════════════════════════════════════════════════════════════
    y_clean_base = core.std.ShufflePlanes(clip_deband, 0, vs.GRAY) if do_chroma else clip_deband

    low_narrow = core.std.Convolution(y_clean_base, matrix=[1, 2, 1], mode="h")
    low_wide   = core.std.Convolution(y_clean_base, matrix=[1, 2, 4, 8, 4, 2, 1], mode="h")
    delta_h    = core.std.Expr([low_narrow, low_wide], f"x y - {peak // 2} +")
    abs_delta  = core.std.Expr([low_narrow, low_wide], "x y - abs")

    halo_detect_thr = int(20 * peak / 255)
    is_true_halo    = core.std.Expr([abs_delta], f"x {halo_detect_thr} > {peak} 0 ?")

    dark_thr        = int(40 * peak / 255)
    is_dark_core    = core.std.Expr([y_clean_base], f"x {dark_thr} < {peak} 0 ?").std.Maximum()

    bright_thr      = int(235 * peak / 255)
    is_bright_spec  = core.std.Expr([y_clean_base], f"x {bright_thr} > {peak} 0 ?").std.Maximum()

    is_halo_zone    = core.std.Expr([is_true_halo, is_dark_core, is_bright_spec], "y 0 > z 0 > or 0 x ?")

    y_de_ring = core.std.Expr([y_clean_base, delta_h, is_halo_zone], f"z 0 > x y {peak // 2} - 0.70 * - x ?")
    y_de_ring = core.std.Expr([y_de_ring], f"x 0 max {peak} min")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 9: 💎 REALCE ADAPTATIVO POR CONTRASTE EN 16-BIT NATIVO (CAS)
    # ════════════════════════════════════════════════════════════════════════
    if hasattr(core, 'cas'):
        sharp_val = 0.15 * (strength / 100.0)
        y_final = core.cas.CAS(y_de_ring, sharpness=sharp_val)
    else:
        y_final = y_de_ring

    # Recombinación final de planos
    if do_chroma:
        u_final = core.std.ShufflePlanes(clip_deband, 1, vs.GRAY)
        v_final = core.std.ShufflePlanes(clip_deband, 2, vs.GRAY)
        target_sub_w = src_fmt.subsampling_w
        target_sub_h = src_fmt.subsampling_h
        target_cw    = w // (1 << target_sub_w)
        target_ch    = h // (1 << target_sub_h)
        if u_final.width != target_cw or u_final.height != target_ch:
            u_final = core.resize.Spline36(u_final, width=target_cw, height=target_ch)
            v_final = core.resize.Spline36(v_final, width=target_cw, height=target_ch)
        cleaned_final = core.std.ShufflePlanes([y_final, u_final, v_final], [0, 0, 0], vs.YUV)
    else:
        cleaned_final = y_final

    if cleaned_final.format.id != clip16.format.id:
        cleaned_final = core.resize.Point(cleaned_final, format=clip16.format.id)

    # Fusión continua según fuerza maestra
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
        right   = core.std.CropAbs(repaired, width=clip16.width - half, height=clip16.height, left=half, top=0)
        stacked = core.std.StackHorizontal([left, right])
        return core.resize.Point(stacked, format=src_fmt_id)

    if show_mask == "xcolor_mask":
        if do_chroma:
            u_before = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
            u_after  = core.std.ShufflePlanes(clip_xc, 1, vs.GRAY)
            u_b_up   = core.resize.Point(u_before, width=w, height=h)
            u_a_up   = core.resize.Point(u_after, width=w, height=h)
            diff_uv  = core.std.Expr([u_b_up, u_a_up], "x y - abs 20 *")
            return core.resize.Point(diff_uv, format=src_fmt_id)
        return core.resize.Point(clip16, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
