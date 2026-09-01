"""
QuesoLimpia — Motor de Restauración Profesional de Polvo y Suciedad
====================================================================
Para VapourBox / VapourSynth R65+

Síntesis de técnicas de las suites más avanzadas del mundo:
  • DVO Dust Fix + Dirt Map + Dry Clean  (Filmworkz / Digital Vision)
  • DIAMANT-Film Restoration Suite       (HS-Art, usado por Library of Congress, BFI)
  • DRS™ Nova                            (MTI Film, ganador del Emmy)

Algoritmo de 8 etapas:
  1. Clasificación de movimiento de cámara (Estático / Bajo / Alto)
  2. Análisis de movimiento jerárquico con MVTools (delta 1 y delta 2)
  3. Detección espacio-temporal multi-rama (polvo brillante, oscuro y de bajo contraste)
  4. Supresión de falsos positivos por grano de película
  5. Refinamiento morfológico + limitador de tamaño (min/max) + protección de bordes
  6. Detección de defectos estáticos (gate hair, suciedad de lente)
  7. Protección de cortes de escena
  8. Reparación selectiva con MaskedMerge + compensación de grano

Controles activables/desactivables:
  • Detección de polvo brillante (detect_bright)
  • Detección de polvo oscuro (detect_dark)
  • Detección de defectos estáticos / gate hair (detect_static)
  • Rama espacial para polvo de bajo contraste (detect_spatial)
  • Supresión de falsos positivos por grano (grain_suppress)
  • Protección de bordes con Canny (edge_protect)
  • Protección de cortes de escena (scene_protect)
  • Compensación de grano en zonas reparadas (grain_restore)
  • Procesamiento de croma (chroma)
  • Modo diagnóstico / visualización de máscara (show_mask)
"""

import vapoursynth as vs

core = vs.core

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------
_MOTION_STATIC = 0
_MOTION_LOW    = 1
_MOTION_HIGH   = 2


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _peak(bits: int) -> int:
    """Valor máximo de píxel para un clip entero de N bits."""
    return (1 << bits) - 1


def _scale(value: float, bits: int) -> float:
    """Escalar un valor de rango 8-bit (0-255) al rango actual."""
    return value * _peak(bits) / 255.0


def _to_workfmt(clip: vs.VideoNode):
    """
    Normalizar a un formato de trabajo seguro:
    - Float → YUV420P16 (los plugins de MVTools aceptan float vía mvsf,
      pero tmedian y ctmf no → convertir)
    - 9-bit → 10-bit  (RestoreMotionBlocks no acepta 9-bit)
    - 4:1:1 → 4:2:2
    Devuelve (clip_trabajado, format_id_original).
    """
    src_fmt = clip.format.id
    bits    = clip.format.bits_per_sample
    family  = clip.format.color_family
    sw      = clip.format.subsampling_w

    needs_conv = (
        clip.format.sample_type == vs.FLOAT
        or bits == 9
        or sw == 2  # 4:1:1
    )

    if needs_conv:
        target_bits = 16 if clip.format.sample_type == vs.FLOAT else (10 if bits == 9 else bits)
        if family == vs.GRAY:
            work_id = core.get_video_format(vs.GRAY8).replace(
                bits_per_sample=target_bits, sample_type=vs.INTEGER).id
        else:
            work_id = core.get_video_format(vs.YUV422P8).replace(
                bits_per_sample=target_bits, sample_type=vs.INTEGER).id
        clip = core.resize.Bicubic(clip, format=work_id)

    return clip, src_fmt


def _restore_fmt(clip: vs.VideoNode, fmt_id: int) -> vs.VideoNode:
    """Devolver al formato original si fue convertido."""
    if clip.format.id != fmt_id:
        return core.resize.Bicubic(clip, format=fmt_id)
    return clip


def _select_mv_funcs(clip: vs.VideoNode):
    """Elegir funciones mv vs mvsf según tipo de muestra."""
    if clip.format.sample_type == vs.FLOAT:
        return core.mvsf.Super, core.mvsf.Analyse, core.mvsf.Compensate, core.mvsf.Recalculate
    return core.mv.Super, core.mv.Analyse, core.mv.Compensate, core.mv.Recalculate


def _auto_blksize(width: int) -> int:
    if width > 2400: return 32
    if width > 960:  return 16
    return 8


def _auto_pel(width: int) -> int:
    return 1 if width > 960 else 2


# ---------------------------------------------------------------------------
# Etapa 1: Clasificación de movimiento de cámara
# (inspirado en MTI Nova "Shine Motion Segmentation")
# ---------------------------------------------------------------------------

def _classify_motion(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Añade la propiedad _QLMotionClass (0=estático, 1=bajo, 2=alto)
    a cada cuadro basándose en la diferencia absoluta media con el siguiente.
    """
    bits    = clip.format.bits_per_sample
    peak_v  = _peak(bits)
    # Clip desplazado un cuadro adelante (el último cuadro se repite)
    shifted = clip[1:] + clip[-1:]
    diff    = core.std.Expr([clip, shifted], "x y - abs")
    diff_s  = core.std.PlaneStats(diff, plane=0)

    def _tag(n, f, c):
        avg   = f.props.get("PlaneStatsAverage", 0)
        level = avg / peak_v  # normalizar a 0-1
        if   level <= 0.005: motion = _MOTION_STATIC
        elif level <= 0.030: motion = _MOTION_LOW
        else:                motion = _MOTION_HIGH
        out = c.std.CopyFrameProps(c)
        out = core.std.SetFrameProps(out, _QLMotionClass=motion)
        return out

    return core.std.FrameEval(clip, lambda n, f: _tag(n, f, clip),
                              prop_src=[diff_s])


# ---------------------------------------------------------------------------
# Etapa 3: Detección espacio-temporal multi-rama
# ---------------------------------------------------------------------------

def _build_dirt_map(
    clip:              vs.VideoNode,
    bc1:               vs.VideoNode,
    fc1:               vs.VideoNode,
    bc2:               vs.VideoNode | None,
    fc2:               vs.VideoNode | None,
    threshold:         float,
    spatial_threshold: float,
    planes:            list[int],
    detect_bright:     bool,
    detect_dark:       bool,
    detect_spatial:    bool,
) -> vs.VideoNode:
    """
    Construye la máscara bruta de suciedad combinando:
      • Rama temporal: intersección bidireccional delta-1 (y delta-2 si hay)
      • Rama espacial: diferencia vs mediana espacial local (bajo contraste)
      • Detección de polvo brillante y oscuro por separado
    """
    bits   = clip.format.bits_per_sample
    peak_v = _peak(bits)
    thr    = threshold
    thr2   = threshold * 0.8  # delta-2 ligeramente más sensible

    luma_plane = [0]

    # — Rama temporal delta-1 —
    # Diferencia absoluta con vecino anterior compensado
    diff_b1 = core.std.Expr([clip, bc1], "x y - abs")
    diff_f1 = core.std.Expr([clip, fc1], "x y - abs")
    # INTERSECCIÓN: alto en AMBAS direcciones → polvo (no movimiento real)
    tmap_d1 = core.std.Expr([diff_b1, diff_f1],
        f"x y min {thr:.4f} > {peak_v} 0 ?")

    # — Rama temporal delta-2 (si está disponible) —
    if bc2 is not None and fc2 is not None:
        diff_b2 = core.std.Expr([clip, bc2], "x y - abs")
        diff_f2 = core.std.Expr([clip, fc2], "x y - abs")
        tmap_d2 = core.std.Expr([diff_b2, diff_f2],
            f"x y min {thr2:.4f} > {peak_v} 0 ?")
        temporal_mask = core.std.Expr([tmap_d1, tmap_d2], "x y max")
    else:
        temporal_mask = tmap_d1

    # — Polvo brillante (punto claro en cuadro actual, oscuro en vecinos) —
    bright_mask = core.std.BlankClip(clip, color=[0] * clip.format.num_planes)
    if detect_bright:
        bright_mask = core.std.Expr(
            [clip, bc1, fc1],
            f"x y z max - {thr * 0.7:.4f} > {peak_v} 0 ?"
        )

    # — Polvo oscuro (punto oscuro en cuadro actual, claro en vecinos) —
    dark_mask = core.std.BlankClip(clip, color=[0] * clip.format.num_planes)
    if detect_dark:
        dark_mask = core.std.Expr(
            [clip, bc1, fc1],
            f"y z min x - {thr * 0.7:.4f} > {peak_v} 0 ?"
        )

    # — Rama espacial (polvo de bajo contraste vs entorno local) —
    spatial_mask = core.std.BlankClip(clip, color=[0] * clip.format.num_planes)
    if detect_spatial:
        local_med    = clip.ctmf.CTMF(radius=2, planes=planes)
        spatial_diff = core.std.Expr([clip, local_med], "x y - abs")
        # Detectar diferencia con la mediana espacial local
        spatial_mask = core.std.Expr([spatial_diff],
            f"x {spatial_threshold:.4f} > {peak_v} 0 ?")

    # — Máscara bruta: unión de todas las ramas —
    raw_mask = core.std.Expr(
        [temporal_mask, bright_mask, dark_mask, spatial_mask],
        "x y max z max a max"
    )

    return raw_mask


# ---------------------------------------------------------------------------
# Etapa 4: Supresión de falsos positivos por grano
# (inspirado en MTI Nova "Grain Suppression Control")
# ---------------------------------------------------------------------------

def _suppress_grain_fps(mask: vs.VideoNode, level: int) -> vs.VideoNode:
    """
    Descarta detecciones que tienen la firma del grano:
    el grano es denso y uniforme → rodea al píxel por todos lados.
    El polvo es aislado → aparece "solo" sin vecinos marcados.

    level: 0-100. 0 = desactivado, 100 = máxima supresión.
    """
    if level == 0:
        return mask
    bits   = mask.format.bits_per_sample
    peak_v = _peak(bits)
    # Si la supresión de grano es moderada, solo suavizar grano disperso
    eroded = mask
    for _ in range(max(1, level // 35)):
        eroded = eroded.std.Minimum(planes=[0])
    for _ in range(max(1, level // 35) + 1):
        eroded = eroded.std.Maximum(planes=[0])
    return core.std.Expr([mask, eroded], "x y min")


# ---------------------------------------------------------------------------
# Etapa 5: Refinamiento morfológico + tamaño + bordes
# ---------------------------------------------------------------------------

def _refine_mask(
    mask:          vs.VideoNode,
    clip:          vs.VideoNode,
    min_size:      int,
    max_size:      int,
    edge_protect:  int,
) -> vs.VideoNode:
    """
    1. Erosión N veces → elimina manchas < min_size
    2. Dilatación N+2 veces → expande las que sobrevivieron (cubre bordes del polvo)
    3. Limitador de tamaño máximo (elimina objetos grandes > max_size)
    4. Protección de bordes con TCanny
    5. Suavizado de bordes de la máscara (Inflate)
    """
    bits   = mask.format.bits_per_sample
    peak_v = _peak(bits)

    m = mask
    # Erosión: descarta manchas muy pequeñas si min_size > 0
    if min_size > 0:
        for _ in range(min_size):
            m = m.std.Minimum(planes=[0])
        for _ in range(min_size + 1):
            m = m.std.Maximum(planes=[0])
    else:
        m = m.std.Maximum(planes=[0])

    # Limitar a max_size: si la mancha es mayor que max_size píxeles,
    # sobrevive a max_size erosiones -> detectarla como objeto grande y excluirla
    if max_size > 0:
        eroded_large = m
        for _ in range(max_size):
            eroded_large = eroded_large.std.Minimum(planes=[0])
        for _ in range(max_size + 2):
            eroded_large = eroded_large.std.Maximum(planes=[0])
        # Excluir objetos grandes de la máscara de suciedad
        m = core.std.Expr([m, eroded_large],
            f"x {peak_v // 2} > y {peak_v // 2} < and {peak_v} 0 ?")

    # Protección de bordes: bordes de alto contraste se protegen
    if edge_protect > 0:
        edges   = clip.tcanny.TCanny(sigma=1.2, mode=0, planes=[0])
        edges_d = edges.std.Maximum(planes=[0])
        edge_threshold = int(peak_v * (101 - edge_protect) / 100 * 0.6)
        m = core.std.Expr([m, edges_d],
            f"x {peak_v // 2} > y {edge_threshold} < and {peak_v} 0 ?")

    # Suavizar bordes de la máscara para transiciones naturales
    m = m.std.Inflate(planes=[0])

    return m


# ---------------------------------------------------------------------------
# Etapa 6: Detección de defectos estáticos (gate hair)
# (inspirado en DIAMANT ExInPaint + MTI Nova Gate Hair Tool)
# ---------------------------------------------------------------------------

def _detect_static_defects(mask: vs.VideoNode) -> vs.VideoNode:
    """
    Un defecto estático es aquel que aparece en el cuadro actual
    Y en el cuadro anterior Y en el cuadro siguiente.
    Inverso al polvo volátil (que solo aparece en UN cuadro).
    """
    bits   = mask.format.bits_per_sample
    peak_v = _peak(bits)
    thr    = peak_v // 2

    mask_prev = mask[-1:] + mask[:-1]   # cuadro anterior
    mask_next = mask[1:]  + mask[-1:]   # cuadro siguiente

    # Marcado en los tres cuadros → defecto estático
    static_mask = core.std.Expr(
        [mask, mask_prev, mask_next],
        f"x {thr} > y {thr} > and z {thr} > and {peak_v} 0 ?"
    )
    # Extraer solo la parte volátil (no estática)
    volatile_mask = core.std.Expr([mask, static_mask],
        f"x {thr} > y {thr} < and {peak_v} 0 ?")

    return volatile_mask, static_mask


# ---------------------------------------------------------------------------
# Etapa 7: Protección de cortes de escena
# (inspirado en DVO Cut Safety + MTI Nova "cut-based safety system")
# ---------------------------------------------------------------------------

def _apply_scene_protection(
    mask:            vs.VideoNode,
    clip:            vs.VideoNode,
    scene_threshold: float,
) -> vs.VideoNode:
    """
    Detecta cortes de escena con misc.SCDetect y anula la máscara
    en el cuadro del corte y en los cuadros adyacentes.
    """
    bits   = mask.format.bits_per_sample
    peak_v = _peak(bits)

    sc = clip.misc.SCDetect(threshold=scene_threshold)

    def _zero_on_cut(n, f, m):
        is_sc = (
            f.props.get("_SceneChangePrev", 0) == 1
            or f.props.get("_SceneChangeNext", 0) == 1
        )
        if is_sc:
            return core.std.BlankClip(m, color=[0] * m.format.num_planes)
        return m

    return core.std.FrameEval(
        mask,
        lambda n, f: _zero_on_cut(n, f, mask),
        prop_src=[sc]
    )


# ---------------------------------------------------------------------------
# Etapa 8a: Reparación selectiva con MaskedMerge
# ---------------------------------------------------------------------------

def _selective_repair(
    clip:           vs.VideoNode,
    bc1:            vs.VideoNode,
    fc1:            vs.VideoNode,
    volatile_mask:  vs.VideoNode,
    static_mask:    vs.VideoNode | None,
    planes:         list[int],
    detect_static:  bool,
) -> vs.VideoNode:
    """
    Polvo volátil → reparar con mediana temporal compensada.
    Defecto estático → reparar con mediana espacial (mismo cuadro).
    Aplicar MaskedMerge para no tocar lo que no está marcado.
    """
    bits   = clip.format.bits_per_sample

    # — Reparación temporal (para polvo volátil) —
    interleaved     = core.std.Interleave([fc1, clip, bc1])
    cleaned_temporal = interleaved.tmedian.TemporalMedian(1, planes)[1::3]
    result = core.std.MaskedMerge(clip, cleaned_temporal, volatile_mask,
                                  planes=planes)

    # — Reparación espacial (para gate hair / defectos estáticos) —
    if detect_static and static_mask is not None:
        cleaned_spatial = clip.ctmf.CTMF(radius=4, planes=planes)
        result = core.std.MaskedMerge(result, cleaned_spatial, static_mask,
                                      planes=planes)

    return result


# ---------------------------------------------------------------------------
# Etapa 8b: Compensación de grano
# (inspirado en DIAMANT grain preservation + MTI Nova grain coherence)
# ---------------------------------------------------------------------------

def _restore_grain(
    original:      vs.VideoNode,
    repaired:      vs.VideoNode,
    volatile_mask: vs.VideoNode,
    static_mask:   vs.VideoNode | None,
    grain_restore: int,
    detect_static: bool,
) -> vs.VideoNode:
    """
    Las zonas reparadas quedan "demasiado limpias" (sin grano).
    Añadir grano sintético calibrado solo en esas zonas para que
    las reparaciones sean invisibles.
    """
    if grain_restore == 0:
        return repaired

    bits   = repaired.format.bits_per_sample
    peak_v = _peak(bits)

    # Calibrar varianza del grano según el nivel solicitado (0-100)
    grain_var_luma   = grain_restore * 0.4
    grain_var_chroma = grain_restore * 0.15

    # Añadir grano al clip reparado
    grained = repaired.grain.Add(
        var=grain_var_luma,
        uvar=grain_var_chroma,
        seed=12345,
        constant=False
    )

    # Máscara combinada: todas las zonas que fueron tocadas
    repair_mask = volatile_mask
    if detect_static and static_mask is not None:
        repair_mask = core.std.Expr([volatile_mask, static_mask], "x y max")

    # Aplicar grano SOLO en zonas reparadas
    return core.std.MaskedMerge(repaired, grained, repair_mask)


# ---------------------------------------------------------------------------
# Función de diagnóstico visual (show_mask)
# ---------------------------------------------------------------------------

def _apply_show_mask(
    clip:          vs.VideoNode,
    repaired:      vs.VideoNode,
    raw_mask:      vs.VideoNode,
    refined_mask:  vs.VideoNode,
    volatile_mask: vs.VideoNode,
    static_mask:   vs.VideoNode | None,
    show_mask:     str,
    detect_static: bool,
) -> vs.VideoNode:
    """
    Modos de visualización:
      "raw"          → mapa de suciedad antes de refinamiento
      "refined"      → máscara después de morfología + tamaño + bordes
      "repair"       → overlay: rojo = polvo volátil, azul = gate hair
      "static"       → solo máscara de defectos estáticos
      "side_by_side" → mitad izquierda original, derecha reparada
    """
    bits   = clip.format.bits_per_sample
    peak_v = _peak(bits)
    w, h   = clip.width, clip.height

    if show_mask == "raw":
        return raw_mask.std.ShufflePlanes(0, vs.GRAY)

    if show_mask == "refined":
        return refined_mask.std.ShufflePlanes(0, vs.GRAY)

    if show_mask == "static" and detect_static and static_mask is not None:
        return static_mask.std.ShufflePlanes(0, vs.GRAY)

    if show_mask == "repair":
        # Overlay rojo en polvo volátil, azul en defectos estáticos
        blank = core.std.BlankClip(clip)
        red   = core.std.MaskedMerge(blank,
            core.std.BlankClip(clip, color=[peak_v, 0, 0]),
            volatile_mask)
        if detect_static and static_mask is not None:
            blue = core.std.MaskedMerge(blank,
                core.std.BlankClip(clip, color=[0, 0, peak_v]),
                static_mask)
            overlay = core.std.Expr([clip, red, blue],
                "x y + z + 3 /")
        else:
            overlay = core.std.Expr([clip, red], "x y + 2 /")
        return overlay

    if show_mask == "side_by_side":
        half   = w // 2
        left   = core.std.CropAbs(clip,    width=half, height=h, left=0, top=0)
        right  = core.std.CropAbs(repaired, width=half, height=h, left=half, top=0)
        return core.std.StackHorizontal([left, right])

    # "off" → devuelve el clip reparado (se llama desde QuesoLimpia)
    return repaired


# ---------------------------------------------------------------------------
# API PÚBLICA
# ---------------------------------------------------------------------------

def QuesoLimpia(
    clip:              vs.VideoNode,
    # Controles generales
    mode:              str   = "balanced",
    strength:          int   = 75,
    # Umbrales de detección
    threshold:         int   = 20,
    spatial_threshold: int   = 15,
    # Controles de tamaño (DVO Dust Size Limiter)
    min_dust_size:     int   = 1,
    max_dust_size:     int   = 16,
    # Ramas de detección activables/desactivables
    detect_bright:     bool  = True,
    detect_dark:       bool  = True,
    detect_spatial:    bool  = True,
    detect_static:     bool  = True,
    # Supresión de grano (MTI Grain Suppression)
    grain_suppress:    int   = 40,
    # Protección de bordes (DVO Speck Reduction)
    edge_protect:      int   = 50,
    # Protección de cortes de escena
    scene_protect:     bool  = True,
    scene_threshold:   float = 0.10,
    # Procesamiento de croma
    chroma:            bool  = True,
    # Compensación de grano en reparaciones
    grain_restore:     int   = 30,
    # Precisión de MVTools
    temporal_radius:   int   = 1,
    blksize:           int | None = None,
    pel:               int | None = None,
    # Diagnóstico visual
    show_mask:         str   = "off",
) -> vs.VideoNode:
    """
    QuesoLimpia — Eliminación profesional de polvo y suciedad temporal.

    Parámetros de control (todos activables/desactivables):
    ─────────────────────────────────────────────────────────
    mode              : "gentle" / "balanced" / "aggressive" / "forensic"
                        Aplica un preset de parámetros. Los valores explícitos
                        que pases sobrescriben el preset.
    strength          : 0-100. Escala global del threshold (75 = normal).
    threshold         : 1-60.  Umbral temporal base (diferencia de píxel).
    spatial_threshold : 1-40.  Umbral para detección espacial (bajo contraste).
    min_dust_size     : 0-8.   Mínimo tamaño en píxeles (descarta grano suelto).
    max_dust_size     : 2-64.  Máximo tamaño en píxeles (evita borrar objetos).

    detect_bright     : True/False. Detectar polvo brillante (puntos blancos).
    detect_dark       : True/False. Detectar polvo oscuro (puntos negros).
    detect_spatial    : True/False. Rama espacial para polvo de bajo contraste.
    detect_static     : True/False. Detectar gate hair / suciedad persistente.
    grain_suppress    : 0-100.  Suprimir falsos positivos por grano de película.
    edge_protect      : 0-100.  Proteger bordes reales (0=desactivado).
    scene_protect     : True/False. Anular máscara en cortes de escena.
    scene_threshold   : 0.01-0.5. Sensibilidad del detector de cortes.
    chroma            : True/False. Procesar planos de croma (U/V).
    grain_restore     : 0-100.  Restaurar grano en zonas reparadas (0=desact.).
    temporal_radius   : 1-2.    1=±1 cuadro, 2=±1 y ±2 cuadros (más preciso).
    blksize           : Tamaño de bloque MVTools (None=auto por resolución).
    pel               : Precisión sub-píxel MVTools (None=auto).

    show_mask         : "off" / "raw" / "refined" / "repair" / "static" / "side_by_side"
                        Modos de diagnóstico visual:
                        "raw"          → mapa de detección antes de refinar
                        "refined"      → máscara después de morfología + bordes
                        "repair"       → overlay: rojo=volátil, azul=estático
                        "static"       → solo detecciones de gate hair
                        "side_by_side" → izquierda original / derecha limpio
    """

    # ── Validación de formato ───────────────────────────────────────────────
    if clip.format is None:
        raise vs.Error("QuesoLimpia: necesita formato de video fijo (no variable).")
    if clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error("QuesoLimpia: solo acepta clips YUV o GRAY.")

    # ── Aplicar modo preconfigurado ─────────────────────────────────────────
    _presets = {
        "gentle":     dict(threshold=30, spatial_threshold=20, max_dust_size=8,
                           min_dust_size=0, grain_suppress=60, edge_protect=80,
                           grain_restore=40),
        "balanced":   dict(threshold=20, spatial_threshold=15, max_dust_size=16,
                           min_dust_size=1, grain_suppress=40, edge_protect=50,
                           grain_restore=30),
        "aggressive": dict(threshold=12, spatial_threshold=10, max_dust_size=32,
                           min_dust_size=1, grain_suppress=25, edge_protect=30,
                           grain_restore=20),
        "forensic":   dict(threshold=8,  spatial_threshold=6,  max_dust_size=48,
                           min_dust_size=0, grain_suppress=15, edge_protect=20,
                           grain_restore=10),
    }
    if mode in _presets:
        p = _presets[mode]
        # Aplicar preset pero respetar valores explícitos del usuario
        # (comparar con los defaults de la firma para detectar si el usuario los cambió)
        _defaults = dict(threshold=20, spatial_threshold=15, max_dust_size=16,
                         min_dust_size=1, grain_suppress=40, edge_protect=50,
                         grain_restore=30)
        if threshold         == _defaults["threshold"]:         threshold         = p["threshold"]
        if spatial_threshold == _defaults["spatial_threshold"]: spatial_threshold = p["spatial_threshold"]
        if max_dust_size     == _defaults["max_dust_size"]:     max_dust_size     = p["max_dust_size"]
        if min_dust_size     == _defaults["min_dust_size"]:     min_dust_size     = p["min_dust_size"]
        if grain_suppress    == _defaults["grain_suppress"]:    grain_suppress    = p["grain_suppress"]
        if edge_protect      == _defaults["edge_protect"]:      edge_protect      = p["edge_protect"]
        if grain_restore     == _defaults["grain_restore"]:     grain_restore     = p["grain_restore"]

    # ── Escalar threshold por strength ──────────────────────────────────────
    strength_scale = strength / 75.0   # 75 = 1.0, 100 = 1.33, 50 = 0.67
    # A mayor strength → umbral MÁS BAJO → detecta más polvo
    threshold         = max(1, threshold / strength_scale)
    spatial_threshold = max(1, spatial_threshold / strength_scale)

    # ── Convertir a formato de trabajo ─────────────────────────────────────
    clip_work, src_fmt_id = _to_workfmt(clip)

    bits    = clip_work.format.bits_per_sample
    is_gray = clip_work.format.color_family == vs.GRAY
    chroma  = False if is_gray else chroma
    planes  = [0, 1, 2] if chroma else [0]

    peak_v = _peak(bits)

    # ── Escalar umbrales al rango de bits actual ────────────────────────────
    thr   = _scale(threshold,         bits)
    s_thr = _scale(spatial_threshold, bits)

    # ── Auto blksize y pel ──────────────────────────────────────────────────
    if blksize is None: blksize = _auto_blksize(clip_work.width)
    if pel     is None: pel     = _auto_pel(clip_work.width)
    overlap = blksize // 2

    # ── ETAPA 1: Clasificación de movimiento ────────────────────────────────
    # (No modifica el clip, solo añade metadatos _QLMotionClass)
    # Nota: El ajuste dinámico de umbral por movimiento se haría en FrameEval,
    # pero para mantener el grafo simple y determinista, lo usamos para
    # información del usuario. Los umbrales fijos son suficientemente robustos
    # gracias al filtro de grano y la intersección bidireccional.
    clip_tagged = _classify_motion(clip_work)

    # ── ETAPA 2: Análisis de movimiento con MVTools ─────────────────────────
    Super, Analyse, Compensate, _ = _select_mv_funcs(clip_work)

    sup = Super(clip_work, pel=pel, sharp=1, rfilter=4)

    bv1 = Analyse(sup, isb=True,  delta=1, blksize=blksize, overlap=overlap, search=5)
    fv1 = Analyse(sup, isb=False, delta=1, blksize=blksize, overlap=overlap, search=5)

    bc1 = Compensate(clip_work, sup, bv1)
    fc1 = Compensate(clip_work, sup, fv1)

    bc2, fc2 = None, None
    if temporal_radius >= 2:
        bv2 = Analyse(sup, isb=True,  delta=2, blksize=blksize, overlap=overlap, search=5)
        fv2 = Analyse(sup, isb=False, delta=2, blksize=blksize, overlap=overlap, search=5)
        bc2 = Compensate(clip_work, sup, bv2)
        fc2 = Compensate(clip_work, sup, fv2)

    # ── ETAPA 3: Detección espacio-temporal ─────────────────────────────────
    raw_mask = _build_dirt_map(
        clip_work, bc1, fc1, bc2, fc2,
        threshold=thr, spatial_threshold=s_thr,
        planes=planes,
        detect_bright=detect_bright,
        detect_dark=detect_dark,
        detect_spatial=detect_spatial,
    )

    # ── ETAPA 4: Supresión de falsos positivos por grano ────────────────────
    if grain_suppress > 0:
        mask_gs = _suppress_grain_fps(raw_mask, grain_suppress)
    else:
        mask_gs = raw_mask

    # ── ETAPA 5: Refinamiento morfológico + tamaño + bordes ─────────────────
    refined_mask = _refine_mask(
        mask_gs, clip_work,
        min_size=min_dust_size,
        max_size=max_dust_size,
        edge_protect=edge_protect if edge_protect > 0 else 0,
    )

    # ── ETAPA 6: Detección de defectos estáticos ────────────────────────────
    if detect_static:
        volatile_mask, static_mask = _detect_static_defects(refined_mask)
    else:
        volatile_mask = refined_mask
        static_mask   = None

    # ── ETAPA 7: Protección de cortes de escena ─────────────────────────────
    if scene_protect:
        volatile_mask = _apply_scene_protection(volatile_mask, clip_work, scene_threshold)
        if static_mask is not None:
            static_mask = _apply_scene_protection(static_mask, clip_work, scene_threshold)

    # ── ETAPA 8a: Reparación selectiva ──────────────────────────────────────
    repaired = _selective_repair(
        clip_work, bc1, fc1,
        volatile_mask=volatile_mask,
        static_mask=static_mask,
        planes=planes,
        detect_static=detect_static,
    )

    # ── ETAPA 8b: Compensación de grano ─────────────────────────────────────
    if grain_restore > 0:
        repaired = _restore_grain(
            original=clip_work,
            repaired=repaired,
            volatile_mask=volatile_mask,
            static_mask=static_mask,
            grain_restore=grain_restore,
            detect_static=detect_static,
        )

    # ── Diagnóstico visual ───────────────────────────────────────────────────
    if show_mask != "off":
        result = _apply_show_mask(
            clip=clip_work,
            repaired=repaired,
            raw_mask=raw_mask,
            refined_mask=refined_mask,
            volatile_mask=volatile_mask,
            static_mask=static_mask,
            show_mask=show_mask,
            detect_static=detect_static,
        )
        return _restore_fmt(result, src_fmt_id)

    # ── Restaurar formato original y devolver ───────────────────────────────
    return _restore_fmt(repaired, src_fmt_id)


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------
quesolimpia = QuesoLimpia
