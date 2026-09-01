/// Parameters for the QuesoLimpia pass.
/// Professional film dust, dirt and spot restoration.
class QuesoLimpiaParameters {
  final bool enabled;
  final String mode;
  final int strength;
  final int threshold;
  final int spatialThreshold;
  final int minDustSize;
  final int maxDustSize;
  final bool detectBright;
  final bool detectDark;
  final bool detectSpatial;
  final bool detectStatic;
  final int grainSuppress;
  final int edgeProtect;
  final bool sceneProtect;
  final double sceneThreshold;
  final bool chroma;
  final int grainRestore;
  final int temporalRadius;
  final int blksize;
  final int pel;
  final String showMask;

  const QuesoLimpiaParameters({
    this.enabled = false,
    this.mode = 'balanced',
    this.strength = 75,
    this.threshold = 20,
    this.spatialThreshold = 15,
    this.minDustSize = 1,
    this.maxDustSize = 16,
    this.detectBright = true,
    this.detectDark = true,
    this.detectSpatial = true,
    this.detectStatic = true,
    this.grainSuppress = 40,
    this.edgeProtect = 50,
    this.sceneProtect = true,
    this.sceneThreshold = 0.10,
    this.chroma = true,
    this.grainRestore = 30,
    this.temporalRadius = 1,
    this.blksize = 16,
    this.pel = 2,
    this.showMask = 'off',
  });

  QuesoLimpiaParameters copyWith({
    bool? enabled,
    String? mode,
    int? strength,
    int? threshold,
    int? spatialThreshold,
    int? minDustSize,
    int? maxDustSize,
    bool? detectBright,
    bool? detectDark,
    bool? detectSpatial,
    bool? detectStatic,
    int? grainSuppress,
    int? edgeProtect,
    bool? sceneProtect,
    double? sceneThreshold,
    bool? chroma,
    int? grainRestore,
    int? temporalRadius,
    int? blksize,
    int? pel,
    String? showMask,
  }) {
    return QuesoLimpiaParameters(
      enabled: enabled ?? this.enabled,
      mode: mode ?? this.mode,
      strength: strength ?? this.strength,
      threshold: threshold ?? this.threshold,
      spatialThreshold: spatialThreshold ?? this.spatialThreshold,
      minDustSize: minDustSize ?? this.minDustSize,
      maxDustSize: maxDustSize ?? this.maxDustSize,
      detectBright: detectBright ?? this.detectBright,
      detectDark: detectDark ?? this.detectDark,
      detectSpatial: detectSpatial ?? this.detectSpatial,
      detectStatic: detectStatic ?? this.detectStatic,
      grainSuppress: grainSuppress ?? this.grainSuppress,
      edgeProtect: edgeProtect ?? this.edgeProtect,
      sceneProtect: sceneProtect ?? this.sceneProtect,
      sceneThreshold: sceneThreshold ?? this.sceneThreshold,
      chroma: chroma ?? this.chroma,
      grainRestore: grainRestore ?? this.grainRestore,
      temporalRadius: temporalRadius ?? this.temporalRadius,
      blksize: blksize ?? this.blksize,
      pel: pel ?? this.pel,
      showMask: showMask ?? this.showMask,
    );
  }

  /// Get a summary string for display.
  String get summary {
    if (!enabled) return 'Off';
    final parts = <String>[];
    parts.add(mode[0].toUpperCase() + mode.substring(1));
    if (detectStatic) parts.add('Gate Hair');
    if (grainRestore > 0) parts.add('Grain ($grainRestore%)');
    return 'QuesoLimpia (${parts.join(", ")})';
  }

  factory QuesoLimpiaParameters.fromJson(Map<String, dynamic> json) {
    return QuesoLimpiaParameters(
      enabled: json['enabled'] as bool? ?? false,
      mode: json['mode'] as String? ?? 'balanced',
      strength: json['strength'] as int? ?? 75,
      threshold: json['threshold'] as int? ?? 20,
      spatialThreshold: json['spatialThreshold'] as int? ?? 15,
      minDustSize: json['minDustSize'] as int? ?? 1,
      maxDustSize: json['maxDustSize'] as int? ?? 16,
      detectBright: json['detectBright'] as bool? ?? true,
      detectDark: json['detectDark'] as bool? ?? true,
      detectSpatial: json['detectSpatial'] as bool? ?? true,
      detectStatic: json['detectStatic'] as bool? ?? true,
      grainSuppress: json['grainSuppress'] as int? ?? 40,
      edgeProtect: json['edgeProtect'] as int? ?? 50,
      sceneProtect: json['sceneProtect'] as bool? ?? true,
      sceneThreshold: (json['sceneThreshold'] as num?)?.toDouble() ?? 0.10,
      chroma: json['chroma'] as bool? ?? true,
      grainRestore: json['grainRestore'] as int? ?? 30,
      temporalRadius: json['temporalRadius'] as int? ?? 1,
      blksize: json['blksize'] as int? ?? 16,
      pel: json['pel'] as int? ?? 2,
      showMask: json['showMask'] as String? ?? 'off',
    );
  }

  Map<String, dynamic> toJson() => {
    'enabled': enabled,
    'mode': mode,
    'strength': strength,
    'threshold': threshold,
    'spatialThreshold': spatialThreshold,
    'minDustSize': minDustSize,
    'maxDustSize': maxDustSize,
    'detectBright': detectBright,
    'detectDark': detectDark,
    'detectSpatial': detectSpatial,
    'detectStatic': detectStatic,
    'grainSuppress': grainSuppress,
    'edgeProtect': edgeProtect,
    'sceneProtect': sceneProtect,
    'sceneThreshold': sceneThreshold,
    'chroma': chroma,
    'grainRestore': grainRestore,
    'temporalRadius': temporalRadius,
    'blksize': blksize,
    'pel': pel,
    'showMask': showMask,
  };
}
