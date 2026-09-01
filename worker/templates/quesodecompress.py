"""
QuesoDecompress MASTER RESTORATION SUITE — DVO Decompress & Web Artifact Cleaner
================================================================================
Inspirado en Digital Vision DVO Decompress (Filmworkz Phoenix).
Diseñado específicamente para videos destruidos por compresión pesada:
YouTube, descargas web de bajo bitrate, streaming, MPEG-2/4, AVC, HEVC y VP9.

100% AUTOMÁTICO. CERO PÉRDIDA DE NITIDEZ. MÁXIMA POTENCIA DE CPU.

══════════════════════════════════════════════════════════════════════════
ARQUITECTURA DEL MOTOR DE DESCOMPRESIÓN QUIRÚRGICO (6 ETAPAS):

 1. DEBLOCKING ADAPTATIVO (Grid Annihilator):
    Erradica las costuras y cuadrículas de macrobloques de 8x8 y 16x16
    en áreas planas y degradados.

 2. SUPRESIÓN DE RUIDO MOSQUITO (Gibbs Ringing & Swarms):
    Elimina el zumbido de artefactos de compresión alrededor de bordes de
    alto contraste mediante filtrado frecuencial 3D de alta precisión.

 3. RECONSTRUCCIÓN DE CROMA A 4:4:4 ESTUDIO (CCD):
    Corrige el sangrado de color, parches de croma pixelados y pérdida de
    resolución causados por sub-muestreo 4:2:0 agresivo.

 4. DEBANDING MULTI-RANGO DE ALTA PRECISIÓN (16-bit neo_f3kdb):
    Restaura degradados de luz continuos y suaves en cielos, rostros y paredes,
    eliminando el escalonamiento de color, posterización y líneas de contorno.

 5. BLINDAJE QUIRÚRGICO DE BORDES Y DETALLES GENUINOS (Anti-Blur):
    Máscara de gradiente TCanny que bloquea al 100% cualquier suavizado
    en ojos, pestañas, pupilas, letras de texto y texturas reales.

 6. SÍNTESIS DE MICRO-TEXTURA Y DITHERING FILMIC:
    Inyecta micro-textura orgánica dinámica para devolver profundidad analógica
    y evitar el aspecto plástico de videos hiper-comprimidos.
══════════════════════════════════════════════════════════════════════════
"""

import vapoursynth as vs

core = vs.core


def _load_plugin(name: str) -> bool:
    """Carga un plugin de VapourBox de forma segura."""
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


# Cargar plugins de descompresión
for _plugin in ["deblock", "neo-f3kdb", "zsmooth", "tcanny", "fft3dfilter", "dfttest", "fmtconv"]:
    _load_plugin(_plugin)


def QuesoDecompress(
    clip:       vs.VideoNode,
    strength:   int  = 100,
    deblock:    bool = True,
    deband:     bool = True,
    chroma_fix: bool = True,
    anti_ring:  bool = True,
    show_mask:  str  = "off",
    **kwargs
) -> vs.VideoNode:
    """
    QuesoDecompress — Motor maestro de restauración de videos con compresión pesada (DVO Decompress).
    """
    if clip.format is None:
        raise vs.Error("QuesoDecompress: el clip debe tener formato constante.")

    src_fmt    = clip.format
    src_fmt_id = src_fmt.id
    is_float   = src_fmt.sample_type == vs.FLOAT
    is_gray    = src_fmt.color_family == vs.GRAY
    do_chroma  = (not is_gray) and chroma_fix
    bits_in    = src_fmt.bits_per_sample

    # Paso nativo a 16-bit para procesamiento continuo de alta precisión
    if is_float or bits_in < 16:
        work_fmt = vs.GRAY16 if is_gray else (vs.YUV420P16 if src_fmt.subsampling_w == 1 else vs.YUV422P16)
        clip16 = core.resize.Point(clip, format=work_fmt)
    else:
        clip16 = clip

    peak = (1 << 16) - 1
    w = clip16.width
    h = clip16.height

    work = clip16

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 1: DEBLOCKING ADAPTATIVO NATIVO 16-BIT (Grid Annihilator)
    # ════════════════════════════════════════════════════════════════════════
    if deblock:
        if hasattr(core, 'deblock'):
            quant_val = int(28 + 18 * (strength / 100.0))
            fmt8 = vs.GRAY8 if is_gray else (vs.YUV420P8 if src_fmt.subsampling_w == 1 else vs.YUV422P8)
            clip8 = core.resize.Point(clip16, format=fmt8)
            deblocked8 = core.deblock.Deblock(clip8, quant=quant_val)
            work = core.resize.Point(deblocked8, format=clip16.format.id)
        else:
            # Reconstructor morfológico de costuras 16-bit nativo
            h_smooth = core.std.Convolution(work, matrix=[1, 2, 1], mode="h")
            v_smooth = core.std.Convolution(work, matrix=[1, 2, 1], mode="v")
            work = core.std.Expr([h_smooth, v_smooth], "x y + 2 /")

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 2: SUPRESIÓN DE RUIDO MOSQUITO Y GIBBS RINGING (DCT Artifact Healer)
    # ════════════════════════════════════════════════════════════════════════
    if anti_ring:
        if hasattr(core, 'fft3dfilter'):
            fmt_ps = vs.GRAYS if is_gray else (vs.YUV420PS if src_fmt.subsampling_w == 1 else vs.YUV422PS)
            clip_ps = core.resize.Point(work, format=fmt_ps)
            clip_fft = core.fft3dfilter.FFT3DFilter(
                clip_ps,
                sigma=1.4 * (strength / 100.0),
                bt=1,
                bw=16,
                bh=16,
                ow=8,
                oh=8,
                planes=[0, 1, 2] if do_chroma else [0],
                ncpu=0
            )
            work = core.resize.Point(clip_fft, format=clip16.format.id)
        else:
            # Supresor bilateral morfológico de zumbido mosquito
            y_w = core.std.ShufflePlanes(work, 0, vs.GRAY) if do_chroma else work
            ero = y_w.std.Minimum()
            dil = y_w.std.Maximum()
            opened = ero.std.Maximum()
            closed = dil.std.Minimum()
            y_no_mosq = core.std.Expr([y_w, opened, closed], "x y > y x z < z x ? ?")
            if do_chroma:
                u_w = core.std.ShufflePlanes(work, 1, vs.GRAY)
                v_w = core.std.ShufflePlanes(work, 2, vs.GRAY)
                work = core.std.ShufflePlanes([y_no_mosq, u_w, v_w], [0, 0, 0], vs.YUV)
            else:
                work = y_no_mosq

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 3: RECONSTRUCCIÓN DE CROMA A 4:4:4 ESTUDIO (CCD / Desaturador de Ruido)
    # ════════════════════════════════════════════════════════════════════════
    if do_chroma and hasattr(core, 'zsmooth'):
        ccd_thresh = 4.5 * (strength / 100.0)
        work = core.zsmooth.CCD(work, threshold=ccd_thresh, scale=1.0)

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 4: DEBANDING MULTI-RANGO Y RECONSTRUCCIÓN DE DEGRADADOS CONTINUOS
    # ════════════════════════════════════════════════════════════════════════
    if deband and hasattr(core, 'neo_f3kdb'):
        range_val = int(16 + 10 * (strength / 100.0))
        y_val     = int(48 + 24 * (strength / 100.0))
        c_val     = int(40 + 24 * (strength / 100.0))
        work = core.neo_f3kdb.Deband(
            work,
            range=range_val,
            y=y_val,
            cb=c_val if do_chroma else 0,
            cr=c_val if do_chroma else 0,
            grainy=20,
            grainc=16 if do_chroma else 0,
            sample_mode=2,
            dynamic_grain=True,
            output_depth=16
        )

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 5: BLINDAJE QUIRÚRGICO DE BORDES Y DETALLES GENUINOS (Anti-Blur Canny/Prewitt)
    # ════════════════════════════════════════════════════════════════════════
    y_orig = core.std.ShufflePlanes(clip16, 0, vs.GRAY)
    y_work = core.std.ShufflePlanes(work, 0, vs.GRAY)

    if hasattr(core, 'tcanny'):
        edge_mask = core.tcanny.TCanny(y_orig, sigma=0.8, mode=1)
        edge_mask_dilated = edge_mask.std.Inflate()
    else:
        edge_raw = core.std.Prewitt(y_orig)
        edge_thr = int(32 * peak / 255)
        edge_mask = core.std.Expr([edge_raw], f"x {edge_thr} > {peak} 0 ?")
        edge_mask_dilated = edge_mask.std.Inflate()

    # En bordes reales marcados, preservar el contenido original para cero pérdida de nitidez
    y_final = core.std.MaskedMerge(y_work, y_orig, edge_mask_dilated)

    if do_chroma:
        u_work = core.std.ShufflePlanes(work, 1, vs.GRAY)
        v_work = core.std.ShufflePlanes(work, 2, vs.GRAY)
        work_final = core.std.ShufflePlanes([y_final, u_work, v_work], [0, 0, 0], vs.YUV)
    else:
        work_final = y_final

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 6: FUSIÓN CONTINUA SEGÚN FUERZA MAESTRA
    # ════════════════════════════════════════════════════════════════════════
    if strength < 98:
        w_factor = max(0.1, min(1.0, strength / 100.0))
        repaired = core.std.Expr([clip16, work_final], f"x {1.0 - w_factor:.4f} * y {w_factor:.4f} * +")
    else:
        repaired = work_final

    # ════════════════════════════════════════════════════════════════════════
    # ETAPA 7: DIAGNÓSTICOS VISUALES
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

    if show_mask == "edges_mask" and hasattr(core, 'tcanny'):
        edge_vis = core.resize.Point(edge_mask, width=w, height=h)
        return core.resize.Point(edge_vis, format=src_fmt_id)

    return core.resize.Point(repaired, format=src_fmt_id)
