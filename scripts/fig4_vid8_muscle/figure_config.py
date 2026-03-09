"""
Shared configuration parameters for muscle activity analysis (videos and figures).
"""

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
BILATERAL_D = 13  # Diameter of pixel neighborhood (0 = compute from sigmaSpace)
BILATERAL_SIGMA_COLOR = 75  # Filter sigma in the color space (larger = more colors mixed)
BILATERAL_SIGMA_SPACE = 75  # Filter sigma in the coordinate space (larger = more distant pixels influence)

# ============================================================================
# IMAGE NORMALIZATION SETTINGS
# ============================================================================
NORM_LOWER_PERCENTILE = 20  # Lower percentile for image normalization
NORM_UPPER_PERCENTILE = 99  # Upper percentile for image normalization

# ============================================================================
# MUSCLE ACTIVITY COMPUTATION
# ============================================================================
TOP_K_PIXELS = 300  # Number of brightest pixels to use for activity computation

# ============================================================================
# SEGMENTATION PROCESSING
# ============================================================================
MORPH_KERNEL_SIZE = 2  # Size of morphological kernel for mask denoising
MORPH_N_ITERATIONS = 1  # Number of morphological iterations
DILATION_KERNEL_SIZE = 1  # Size of dilation kernel for final masks

# ============================================================================
# BASELINE AND DELTA F/F0 COMPUTATION
# ============================================================================
BASELINE_WINDOW_SEC = 0.5  # Duration (seconds) before/after stimulation for baseline

# ============================================================================
# OVERLAY VISUALIZATION SETTINGS
# ============================================================================
SEGMENT_BLEND_ORIGINAL = 0.6  # Weight of original image in segment overlay
SEGMENT_BLEND_COLOR = 0.4  # Weight of segment color in overlay
TOP_K_HIGHLIGHT_FACTOR = 0.5  # Brightness factor for top-k pixel highlighting

# ============================================================================
# SEGMENT COLOR PALETTE
# ============================================================================
SEGMENT_COLORS = {
    "LFFemur": "#e41a1c",
    "LMFemur": "#377eb8",
    "LHFemur": "#4daf4a",
    "RFFemur": "#984ea3",
    "RMFemur": "#ff7f00",
    "RHFemur": "#fdbf6f",  # Changed from yellow to light orange
    "LFTibia": "#a65628",
    "LMTibia": "#f781bf",
    "LHTibia": "#999999",
    "RFTibia": "#66c2a5",
    "RMTibia": "#fc8d62",
    "RHTibia": "#8da0cb",
}

# ============================================================================
# TRACE PLOT STYLING
# ============================================================================
TRACE_LINEWIDTH = 1.5
TRACE_BASELINE_ALPHA = 0.3
STIM_PERIOD_ALPHA = 0.2

# ============================================================================
# VIDEO-SPECIFIC SETTINGS
# ============================================================================
OUTPUT_CODEC = "libx264"
CODEC_PRESET = "veryfast"
CODEC_CRF = "18"
OUTPUT_FPS = 33
TRACE_PANEL_HEIGHT_RATIO = 2/3
TRACE_CURRENT_TIME_WIDTH = 2
TRACE_CURRENT_TIME_ALPHA = 0.8

# ============================================================================
# STATIC FIGURE-SPECIFIC SETTINGS
# ============================================================================
STIMULUS_INDEX = 0  # Which stimulus to visualize (0 = first, 1 = second, etc.)
FRAME_OFFSETS = [0, 3]  # Frame offsets from stimulus onset
FIGURE_DPI = 500
TRACE_FIGURE_WIDTH = 10
TRACE_FIGURE_HEIGHT = 3
ANNOTATION_FONT_SIZE = 24
ANNOTATION_COLOR = (255, 255, 255)
ANNOTATION_POSITION = (20, 40)
STIM_PERIOD_COLOR = "#D3D3D3"  # Light gray for stimulation periods
