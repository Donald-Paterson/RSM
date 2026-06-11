import os

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

# =========================================================
# ANNOTATION TYPE
# =========================================================

ANNOTATION_TYPE = "XML"

# OPTIONS:
# "YOLO"
# "XML"

# =========================================================
# RSM MODE
# =========================================================

RSM_MODE = "STRUCTURE"

# OPTIONS:
# "STRUCTURE"
# "DEFECT"

# =========================================================
# QUADRANT MODE
# =========================================================

ENABLE_QUADRANTS = True

# =========================================================
# PREPROCESSING
# =========================================================

ENABLE_PREPROCESSING = True

SAVE_PREPROCESSED = True

# =========================================================
# PATHS
# =========================================================

PROJECT_NAME = "Indomim-side-p1"

PROJECT_FOLDER = os.path.join(
    BASE_DIR,
    "Data",
    PROJECT_NAME
)

REFERENCE_IMAGE = os.path.join(
    PROJECT_FOLDER,
    "images",
    "pos_1.png"
)

# =========================================================
# LABEL PATH
# =========================================================

if ANNOTATION_TYPE == "YOLO":

    REFERENCE_LABEL = os.path.join(
        PROJECT_FOLDER,
        "labels",
        "golden.txt"
    )

else:

    REFERENCE_LABEL = os.path.join(
        PROJECT_FOLDER,
        "labels",
        "annotations.xml"
    )

# =========================================================
# TEST IMAGES
# =========================================================

TEST_FOLDER = os.path.join(
    BASE_DIR,
    "Data",
    "test_images",
    PROJECT_NAME
)

# =========================================================
# OUTPUT FOLDERS
# =========================================================

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs"
)

DEBUG_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "debug"
)

ROI_DEBUG_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "roi_debug"
)

REPORT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "Reports"
)

REPORT_PATH = os.path.join(
    REPORT_FOLDER,
    "report.xlsx"
)

PREPROCESS_FOLDER = os.path.join(
    BASE_DIR,
    "Data",
    "preprocessed"
)

# =========================================================
# AUTO CREATE FOLDERS
# =========================================================

os.makedirs(
    DEBUG_FOLDER,
    exist_ok=True
)

os.makedirs(
    ROI_DEBUG_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    PREPROCESS_FOLDER,
    exist_ok=True
)

# =========================================================
# ALIGNMENT
# =========================================================

ENABLE_ALIGNMENT = True

# =========================================================
# THRESHOLDS
# =========================================================

FINAL_PASS_SCORE = 0.65

ABSENCE_SSIM_THRESHOLD = 0.40

MAX_DEFECT_RATIO = 0.01

DIFF_THRESHOLD = 35

MIN_BLOB_AREA = 20

# =========================================================
# STRUCTURE MODE THRESHOLDS
# =========================================================

SSIM_PASS_THRESHOLD = 0.80

EDGE_PASS_THRESHOLD = 0.15

# =========================================================
# DEFECT MODE THRESHOLDS
# =========================================================

LOCAL_SIMILARITY_THRESHOLD = 0.98

# =========================================================
# PREPROCESSING PARAMETERS
# =========================================================

PREPROCESS_BRIGHTNESS = 30

PREPROCESS_CONTRAST = 3.0

# =========================================================
# METRIC WEIGHTS
# =========================================================

W_SSIM = 0.25

W_EDGE = 0.25

W_MSE = 0.15

W_HIST = 0.20

W_GRAD = 0.15

# =========================================================
# ROI-SPECIFIC WEIGHTS
# =========================================================

ROI_METRIC_WEIGHTS = {

    0: {

        "ssim": 0.20,

        "edge": 0.30,

        "mse": 0.15,

        "hist": 0.20,

        "grad": 0.15
    }
}