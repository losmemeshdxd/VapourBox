use serde::{Deserialize, Serialize};

/// Parameters for the QuesoLimpia pass.
/// Professional film dust, dirt and spot restoration with motion-aware spatio-temporal detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuesoLimpiaParameters {
    /// Whether this pass is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Restoration mode preset: "gentle", "balanced", "aggressive", "forensic"
    #[serde(default = "default_mode")]
    pub mode: String,

    /// General strength scale (10-100, default 75)
    #[serde(default = "default_strength")]
    pub strength: i32,

    /// Temporal base threshold (1-60, default 20)
    #[serde(default = "default_threshold")]
    pub threshold: i32,

    /// Spatial threshold for low-contrast dirt (1-40, default 15)
    #[serde(default = "default_spatial_threshold")]
    pub spatial_threshold: i32,

    /// Minimum dust spot size in pixels (0-8, default 1)
    #[serde(default = "default_min_dust_size")]
    pub min_dust_size: i32,

    /// Maximum dust spot size in pixels (2-64, default 16)
    #[serde(default = "default_max_dust_size")]
    pub max_dust_size: i32,

    /// Detect bright dust spots (default true)
    #[serde(default = "default_true")]
    pub detect_bright: bool,

    /// Detect dark dust spots (default true)
    #[serde(default = "default_true")]
    pub detect_dark: bool,

    /// Detect low-contrast spatial spots (default true)
    #[serde(default = "default_true")]
    pub detect_spatial: bool,

    /// Detect static defects / gate hair (default true)
    #[serde(default = "default_true")]
    pub detect_static: bool,

    /// Grain suppression to avoid false positives (0-100, default 40)
    #[serde(default = "default_grain_suppress")]
    pub grain_suppress: i32,

    /// Edge protection with Canny (0-100, default 50)
    #[serde(default = "default_edge_protect")]
    pub edge_protect: i32,

    /// Scene cut protection (default true)
    #[serde(default = "default_true")]
    pub scene_protect: bool,

    /// Scene cut sensitivity threshold (default 0.10)
    #[serde(default = "default_scene_threshold")]
    pub scene_threshold: f64,

    /// Process chroma planes (default true)
    #[serde(default = "default_true")]
    pub chroma: bool,

    /// Grain compensation in repaired areas (0-100, default 30)
    #[serde(default = "default_grain_restore")]
    pub grain_restore: i32,

    /// Temporal radius: 1=±1 frame, 2=±2 frames (default 1)
    #[serde(default = "default_temporal_radius")]
    pub temporal_radius: i32,

    /// Block size for motion analysis (default 16)
    #[serde(default = "default_blksize")]
    pub blksize: i32,

    /// Sub-pixel accuracy (default 2)
    #[serde(default = "default_pel")]
    pub pel: i32,

    /// Diagnostic visualization mode: "off", "raw", "refined", "repair", "static", "side_by_side"
    #[serde(default = "default_show_mask")]
    pub show_mask: String,
}

fn default_true() -> bool { true }
fn default_mode() -> String { "balanced".to_string() }
fn default_strength() -> i32 { 75 }
fn default_threshold() -> i32 { 20 }
fn default_spatial_threshold() -> i32 { 15 }
fn default_min_dust_size() -> i32 { 1 }
fn default_max_dust_size() -> i32 { 16 }
fn default_grain_suppress() -> i32 { 40 }
fn default_edge_protect() -> i32 { 50 }
fn default_scene_threshold() -> f64 { 0.10 }
fn default_grain_restore() -> i32 { 30 }
fn default_temporal_radius() -> i32 { 1 }
fn default_blksize() -> i32 { 16 }
fn default_pel() -> i32 { 2 }
fn default_show_mask() -> String { "off".to_string() }

impl Default for QuesoLimpiaParameters {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: default_mode(),
            strength: default_strength(),
            threshold: default_threshold(),
            spatial_threshold: default_spatial_threshold(),
            min_dust_size: default_min_dust_size(),
            max_dust_size: default_max_dust_size(),
            detect_bright: true,
            detect_dark: true,
            detect_spatial: true,
            detect_static: true,
            grain_suppress: default_grain_suppress(),
            edge_protect: default_edge_protect(),
            scene_protect: true,
            scene_threshold: default_scene_threshold(),
            chroma: true,
            grain_restore: default_grain_restore(),
            temporal_radius: default_temporal_radius(),
            blksize: default_blksize(),
            pel: default_pel(),
            show_mask: default_show_mask(),
        }
    }
}
