from pathlib import Path

import cv2
import numpy as np

from app.backend.config import settings
from app.backend.schemas import DefectBoundingBox


def save_roi_debug(
    position: str,
    image_name: str,
    roi_index: int,
    quadrant: int,
    golden_roi: np.ndarray,
    test_roi: np.ndarray,
    diff_map: np.ndarray | None = None,
) -> tuple[str, str, str]:
    image_stem = Path(image_name).stem
    folder = settings.debug_root / position / image_stem
    folder.mkdir(parents=True, exist_ok=True)

    suffix = f"roi_{roi_index}_Q{quadrant}"
    golden_path = folder / f"{suffix}_golden.png"
    test_path = folder / f"{suffix}_test.png"
    diff_path = folder / f"{suffix}_diff.png"

    cv2.imwrite(str(golden_path), golden_roi)
    cv2.imwrite(str(test_path), test_roi)

    if diff_map is not None:
        diff_image = diff_map if len(diff_map.shape) == 2 else cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    else:
        diff_image = cv2.absdiff(golden_roi, test_roi)

    cv2.imwrite(str(diff_path), diff_image)

    return str(golden_path), str(test_path), str(diff_path)


def save_annotated_test_image(
    position: str,
    image_name: str,
    aligned_test_image: np.ndarray,
    bounding_boxes: list[DefectBoundingBox],
) -> Path:
    folder = settings.debug_root / position / Path(image_name).stem
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / "aligned_test_with_defects.png"

    annotated = aligned_test_image.copy()
    image_height, image_width = annotated.shape[:2]
    visual_padding = 8

    for box in bounding_boxes:
        x_min = max(0, box.x_min - visual_padding)
        y_min = max(0, box.y_min - visual_padding)
        x_max = min(image_width - 1, box.x_max - 1 + visual_padding)
        y_max = min(image_height - 1, box.y_max - 1 + visual_padding)

        cv2.rectangle(
            annotated,
            (x_min, y_min),
            (x_max, y_max),
            (0, 0, 255),
            2,
        )

    cv2.imwrite(str(output_path), annotated)
    return output_path




