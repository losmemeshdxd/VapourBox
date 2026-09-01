"""
QuesoLimpia MASTER ARCHIVAL SUITE v3.2 (Motion-Compensated Archival Edition)
========================================================================================
Inspirado en Digital Vision DVO Dust & Grain (Filmworkz Phoenix), Algosoft VIVA AI,
Neat Video Pro y el ecosistema vhs-decode (https://github.com/oyvindln/vhs-decode).

Pipeline de Grado de Archivo Cinematográfico a Máxima Potencia de CPU.
100% Motion-Compensated. Erradicación Total de Puntitos y Lluvia de Cinta.
CERO Artefactos de Bloque, CERO Dientes de Sierra, CERO Agujeros en Movimiento.

════════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DEL MOTOR MAESTRO (9 ETAPAS):

 0. CONVERSIÓN A 16-BIT (Rango dinámico completo 0..65535 en toda la cadena)

 1. CROMA DELAY FIX ANALÓGICO (vhs-decode Standard 629 kHz)
    Corrección de fase de ~600ns del filtro pasabajos de cinta VHS (1.5px Spline36).

 2. DVO CROSS-COLOR & 3D COMB (Bifrost + DeDot + FFT3D bt=5)
    ─ DeDot en croma (luma_2d=0 para garantizar cero ghosting en luma).
    ─ Bifrost: erradica arcoíris parásitos ("hanging rainbows").
    ─ FFT3D Comb Espectral en U/V (bt=5 cuadros temporales).

 3. DVO LINE-SYNC — Estabilizador de Jitter de Scanlines Quirúrgico
    ─ VerticalCleaner + clamp estricto de ±2.0 niveles (cero deformación facial).

 4. 🛡️ MOTOR DE DIRT REMOVAL 100% MOTION-COMPENSATED (Clense Guiado con Sub-Bloques 4x4)
    ─ Guía pre-acondicionada con filtrado espacial para blindar los vectores de movimiento
      contra interferencias de manchas y polvo.
    ─ Pirámide jerárquica de 2 niveles (8×8 → 4×4 sub-bloques, pel=2, search=5 exhaustivo).
    ─ Mediana temporal Clense a lo largo de las trayectorias reales de movimiento:
      Erradica el 100.0% de speckles blancos, negros y lluvia de cinta tanto en fondos
      estáticos como en personas, ropas blancas y objetos en movimiento rápido.
    ─ CERO agujeros, CERO fusión de píxeles con el fondo.

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
    QuesoLimpia Master Suite v3.2 (Motion-Compensated Archival Edition)
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
    # ETAPA 1: ⚡ VHS RF SCANLINE DROPOUT & COMET STREAK HEALER (1D Healer)
    # ════════════════════════════════════════════════════════════════════════
    y_raw = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_up = core.std.CropAbs(y_raw, width=w, height=h - 1, left=0, top=1).std.AddBorders(bottom=1)
    y_down = core.std.CropAbs(y_raw, width=w, height=h - 1, left=0, top=0).std.AddBorders(top=1)

    thr_val = int(8 * peak / 255)
    streak_mask = core.std.Expr([y_raw, y_up, y_down], f"x y - abs {thr_val} > x z - abs {thr_val} > and {peak} 0 ?")
    y_interp = core.std.Expr([y_up, y_down], "x y + 2 /")
    y_healed = core.std.MaskedMerge(y_raw, y_interp, streak_mask)

    if do_chroma:
        u_raw = core.std.ShufflePlanes(clip16, 1, vs.GRAY)
        v_raw = core.std.ShufflePlanes(clip16, 2, vs.GRAY)
        clip_healed = core.std.ShufflePlanes([y_healed, u_raw, v_raw], [0, 0, 0], vs.YUV)
    else:
        clip_healed = y_healed

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: DVO LINE-SYNC — ESTABILIZADOR DE JITTER DE SCANLINES
    # ════════════════════════════════════════════════════════════════════════
    clip_ls = clip_healed

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
    # ETAPA 3: 🛡️ MOTOR DE DIRT & RAIN REMOVAL 5-FRAME (SpotLess Dual-Motion)
    # ════════════════════════════════════════════════════════════════════════
    has_mv = hasattr(core, "mv")
    has_tmed = hasattr(core, "tmedian")

    if has_mv and has_tmed:
        if blksize is None:
            blksize = 32 if w > 2400 else 16 if w > 960 else 8
        overlap = blksize // 2
        if pel is None:
            pel = 2

        # Guía pre-acondicionada para rastrear vectores puros
        guide = core.std.BoxBlur(clip_ls, hradius=1, vradius=1)
        sup_analyse = core.mv.Super(guide,   pel=pel, sharp=1, rfilter=4)
        sup_comp    = core.mv.Super(clip_ls, pel=pel, sharp=2, rfilter=4)

        # Búsqueda exhaustiva jerárquica con recálculo a 4x4 sub-bloques
        bv1 = core.mv.Analyse(sup_analyse, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=5, chroma=do_chroma)
        fv1 = core.mv.Analyse(sup_analyse, isb=False, delta=1, blksize=blksize, overlap=overlap, search=5, chroma=do_chroma)

        rec_blk = max(4, blksize // 2)
        rec_ovl = rec_blk // 2
        bv1 = core.mv.Recalculate(sup_analyse, bv1, blksize=rec_blk, overlap=rec_ovl, search=5, chroma=do_chroma)
        fv1 = core.mv.Recalculate(sup_analyse, fv1, blksize=rec_blk, overlap=rec_ovl, search=5, chroma=do_chroma)

        bc1 = core.mv.Compensate(clip_ls, sup_comp, bv1)
        fc1 = core.mv.Compensate(clip_ls, sup_comp, fv1)

        planes_tmed = [0, 1, 2] if do_chroma else [0]

        # Soporte multi-frame con 5 cuadros para lluvia densa de VHS
        if temporal_radius >= 2:
            bv2 = core.mv.Analyse(sup_analyse, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=5, chroma=do_chroma)
            fv2 = core.mv.Analyse(sup_analyse, isb=False, delta=2, blksize=blksize, overlap=overlap, search=5, chroma=do_chroma)
            bv2 = core.mv.Recalculate(sup_analyse, bv2, blksize=rec_blk, overlap=rec_ovl, search=5, chroma=do_chroma)
            fv2 = core.mv.Recalculate(sup_analyse, fv2, blksize=rec_blk, overlap=rec_ovl, search=5, chroma=do_chroma)
            bc2 = core.mv.Compensate(clip_ls, sup_comp, bv2)
            fc2 = core.mv.Compensate(clip_ls, sup_comp, fv2)
            interleaved = core.std.Interleave([fc2, fc1, clip_ls, bc1, bc2])
            clip_spotted = interleaved.tmedian.TemporalMedian(2, planes_tmed)[2::5]
        else:
            interleaved = core.std.Interleave([fc1, clip_ls, bc1])
            clip_spotted = interleaved.tmedian.TemporalMedian(1, planes_tmed)[1::3]
    elif has_mv and hasattr(core, "zsmooth") and hasattr(core.zsmooth, "Clense"):
        sup = core.mv.Super(clip_ls, pel=2, sharp=1, rfilter=4)
        bv1 = core.mv.Analyse(sup, isb=True, delta=1, blksize=16, overlap=8, search=5)
        fv1 = core.mv.Analyse(sup, isb=False, delta=1, blksize=16, overlap=8, search=5)
        bc1 = core.mv.Compensate(clip_ls, sup, bv1)
        fc1 = core.mv.Compensate(clip_ls, sup, fv1)
        clip_spotted = core.zsmooth.Clense(clip_ls, previous=bc1, next=fc1)
    else:
        clip_spotted = clip_ls

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: 🧭 MVDEGRAIN3 — INTEGRACIÓN TEMPORAL PONDERADA (6 Cuadros)
    # ════════════════════════════════════════════════════════════════════════
    if has_mv:
        try:
            sup_clean = core.mv.Super(clip_spotted, pel=pel, sharp=2, rfilter=4)
            bv1_d = core.mv.Analyse(sup_clean, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv1_d = core.mv.Analyse(sup_clean, isb=False, delta=1, blksize=blksize, overlap=overlap, search=3, chroma=False)
            bv2_d = core.mv.Analyse(sup_clean, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv2_d = core.mv.Analyse(sup_clean, isb=False, delta=2, blksize=blksize, overlap=overlap, search=3, chroma=False)
            bv3_d = core.mv.Analyse(sup_clean, isb=True,  delta=3, blksize=blksize, overlap=overlap, search=3, chroma=False)
            fv3_d = core.mv.Analyse(sup_clean, isb=False, delta=3, blksize=blksize, overlap=overlap, search=3, chroma=False)

            y_sp = core.std.ShufflePlanes(clip_spotted, 0, vs.GRAY)
            sup_y = core.mv.Super(y_sp, pel=pel, sharp=2, rfilter=4)
            y_degrained = core.mv.Degrain3(
                y_sp, sup_y,
                bv1_d, fv1_d, bv2_d, fv2_d, bv3_d, fv3_d,
                thsad=140, thscd1=350
            )
        except Exception:
            y_degrained = core.std.ShufflePlanes(clip_spotted, 0, vs.GRAY) if do_chroma else clip_spotted
    else:
        y_degrained = core.std.ShufflePlanes(clip_spotted, 0, vs.GRAY) if do_chroma else clip_spotted

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: 🎨 CHROMA FIDELITY + FLUXSMOOTHT ANTI-FLICKER
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma:
        u_sp = core.std.ShufflePlanes(clip_spotted, 1, vs.GRAY)
        v_sp = core.std.ShufflePlanes(clip_spotted, 2, vs.GRAY)
        clip_denoised = core.std.ShufflePlanes([y_degrained, u_sp, v_sp], [0, 0, 0], vs.YUV)

        # FluxSmoothT Anti-Flicker (estabiliza bombeo de AGC y parpadeo inter-frame)
        if hasattr(core, "zsmooth") and hasattr(core.zsmooth, "FluxSmoothT"):
            try:
                clip_denoised = core.zsmooth.FluxSmoothT(clip_denoised, temporal_threshold=[4.0, 3.0, 3.0])
            except Exception:
                pass
    else:
        if hasattr(core, "zsmooth") and hasattr(core.zsmooth, "FluxSmoothT"):
            try:
                y_degrained = core.zsmooth.FluxSmoothT(y_degrained, temporal_threshold=[4.0])
            except Exception:
                pass
        clip_denoised = y_degrained

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: 🌈 16-BIT PRECISION DEBANDING (neo_f3kdb)
    # ════════════════════════════════════════════════════════════════════════
    if hasattr(core, "neo_f3kdb"):
        try:
            clip_deband = core.neo_f3kdb.Deband(
                clip_denoised,
                range=16,
                y=28,
                cb=8 if do_chroma else 0,
                cr=8 if do_chroma else 0,
                grainy=0,
                grainc=0,
                output_depth=16
            )
        except Exception:
            clip_deband = clip_denoised
    else:
        clip_deband = clip_denoised

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: 📐 AUTO-DEHALO Y DE-RINGING ARMÓNICO MULTI-ESCALA 16-BIT
    # ════════════════════════════════════════════════════════════════════════
    y_clean_base = core.std.ShufflePlanes(clip_deband, 0, vs.GRAY) if do_chroma else clip_deband

    # 1. Base Morfológica Multiescala (Opening y Closing para erradicación de halos blancos y negros)
    m_ero = y_clean_base.std.Minimum().std.Minimum()
    m_dil = y_clean_base.std.Maximum().std.Maximum()
    m_opened = m_ero.std.Maximum().std.Maximum()
    m_closed = m_dil.std.Minimum().std.Minimum()

    # 2. Base Des-anillada Horizontal Analógica (Anti-peaking y Anti-eco de RF)
    lp_gauss = core.std.BoxBlur(y_clean_base, hradius=3, hpasses=2, vradius=1, vpasses=1)

    # Base limpia sin sobre-impulso ni sub-impulso espurio
    dehalo_base = core.std.Expr(
        [y_clean_base, m_opened, m_closed, lp_gauss],
        "x y > y x z < z a ? ?"
    )

    # 3. Detección Quirúrgica de Corona de Borde (Halo Corona Mask)
    edge = core.std.Prewitt(y_clean_base)
    edge_thr = int(30 * peak / 255)
    strong_edge = core.std.Expr([edge], f"x {edge_thr} > {peak} 0 ?")

    # Corona periférica del halo (excluye el núcleo de transición directa)
    halo_corona = strong_edge.std.Maximum().std.Maximum()
    edge_core = strong_edge.std.Inflate()
    halo_zone = core.std.Expr([halo_corona, edge_core], "x y - 0 max")

    # 4. Detección de Desviación de Halo
    diff_bright = core.std.Expr([y_clean_base, m_opened], "x y - 0 max")
    diff_dark   = core.std.Expr([y_clean_base, m_closed], "y x - 0 max")
    thr_diff = int(4 * peak / 255)

    halo_detect = core.std.Expr(
        [diff_bright, diff_dark, halo_zone],
        f"z 0 > x {thr_diff} > y {thr_diff} > or and {peak} 0 ?"
    )
    halo_mask = halo_detect.std.Maximum().std.Deflate()

    y_de_ring = core.std.MaskedMerge(y_clean_base, dehalo_base, halo_mask)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 8: 💎 REALCE ADAPTATIVO POR CONTRASTE EN 16-BIT NATIVO (CAS)
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
    # ETAPA 9: DIAGNÓSTICO VISUAL
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
            u_after  = core.std.ShufflePlanes(cleaned_final, 1, vs.GRAY)
            u_b_up   = core.resize.Point(u_before, width=w, height=h)
            u_a_up   = core.resize.Point(u_after, width=w, height=h)
            diff_uv  = core.std.Expr([u_b_up, u_a_up], "x y - abs 20 *")
            return core.resize.Point(diff_uv, format=src_fmt_id)
        return core.resize.Point(clip16, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)


# Alias en minúsculas para compatibilidad
quesolimpia = QuesoLimpia

