from pathlib import Path

import cv2

from app.backend.config import settings
from app.backend.schemas import DefectBoundingBox, RoiResult
from app.vision.alignment import align_images
from app.vision.debug_writer import save_annotated_test_image, save_roi_debug
from app.vision.decision_engine import compute_score, decide_structure
from app.vision.image_utils import load_image
from app.vision.models import PipelineOutput
from app.vision.preprocessing import preprocess_roi
from app.vision.roi_extractor import divide_into_quadrants, extract_polygon_rois, load_annotations
from app.vision.similarity import (
    best_shift_match,
    edge_difference,
    gradient_difference,
    histogram_similarity,
    local_difference_analysis,
    mse_score,
)


class RSMPipeline:
    """Backend adapter for the established RSM validator sequence."""

    def run(
        self,
        project_id: str,
        position: str,
        golden_image_path: Path,
        test_image_path: Path,
        annotation_path: Path,
    ) -> PipelineOutput:
        golden = load_image(golden_image_path)
        test = load_image(test_image_path)
        aligned = align_images(golden, test) if settings.enable_alignment else test

        components = load_annotations(annotation_path)
        golden_rois = extract_polygon_rois(golden, components)
        test_rois = extract_polygon_rois(aligned, components)

        results = []
        image_defect_boxes = []
        artifact_scope = f"{project_id}/{position}"

        for roi_index, (golden_data, test_data) in enumerate(zip(golden_rois, test_rois)):
            golden_roi = golden_data["roi"]
            test_roi = test_data["roi"]

            if golden_roi.shape != test_roi.shape:
                test_roi = cv2.resize(test_roi, (golden_roi.shape[1], golden_roi.shape[0]))

            golden_parts = divide_into_quadrants(golden_roi) if settings.enable_quadrants else [golden_roi]
            test_parts = divide_into_quadrants(test_roi) if settings.enable_quadrants else [test_roi]

            for quadrant_index, (golden_part, test_part) in enumerate(zip(golden_parts, test_parts)):
                if golden_part.size == 0 or test_part.size == 0:
                    continue

                if golden_part.shape != test_part.shape:
                    test_part = cv2.resize(test_part, (golden_part.shape[1], golden_part.shape[0]))

                quadrant = quadrant_index + 1

                if settings.enable_preprocessing:
                    preprocess_folder = settings.preprocess_root / artifact_scope / test_image_path.stem
                    golden_part = preprocess_roi(
                        golden_part,
                        save=settings.save_preprocessed,
                        save_path=preprocess_folder / f"{test_image_path.name}_roi{roi_index}_Q{quadrant}_golden.png",
                    )
                    test_part = preprocess_roi(
                        test_part,
                        save=settings.save_preprocessed,
                        save_path=preprocess_folder / f"{test_image_path.name}_roi{roi_index}_Q{quadrant}_test.png",
                    )

                result = self._inspect_roi(
                    project_id=project_id,
                    position=position,
                    artifact_scope=artifact_scope,
                    image_name=test_image_path.name,
                    roi_index=roi_index,
                    quadrant=quadrant,
                    label=golden_data["label"],
                    roi_bbox=golden_data["bbox"],
                    full_roi_shape=golden_roi.shape,
                    golden_roi=golden_part,
                    test_roi=test_part,
                )
                results.append(result)
                image_defect_boxes.extend(result.defect_bounding_boxes)

        annotated_path = save_annotated_test_image(
            position=artifact_scope,
            image_name=test_image_path.name,
            aligned_test_image=aligned,
            bounding_boxes=image_defect_boxes,
        )

        return PipelineOutput(
            results=results,
            defect_bounding_boxes=image_defect_boxes,
            aligned_annotated_test_image_path=annotated_path,
        )

    def _inspect_roi(
        self,
        project_id: str,
        position: str,
        artifact_scope: str,
        image_name: str,
        roi_index: int,
        quadrant: int,
        label: str,
        roi_bbox: tuple[int, int, int, int],
        full_roi_shape,
        golden_roi,
        test_roi,
    ) -> RoiResult:
        local_similarity, defect_ratio, _, local_boxes = local_difference_analysis(golden_roi, test_roi)

        if settings.rsm_mode == "STRUCTURE":
            ssim_value = best_shift_match(golden_roi, test_roi)
            edge_value = edge_difference(golden_roi, test_roi)
            mse_value = mse_score(golden_roi, test_roi)
            hist_value = histogram_similarity(golden_roi, test_roi)
            grad_value = gradient_difference(golden_roi, test_roi)

            final_score = compute_score(
                ssim_value,
                edge_value,
                mse_value,
                hist_value,
                grad_value,
                roi_index,
            )
            prediction = decide_structure(final_score, ssim_value, edge_value, defect_ratio)
        else:
            ssim_value = local_similarity
            edge_value = defect_ratio
            mse_value = 0.0
            hist_value = 0.0
            grad_value = 0.0
            final_score = local_similarity
            prediction = "FAIL" if defect_ratio > settings.max_defect_ratio else "PASS"

        defect_boxes = []
        if prediction != "PASS":
            defect_boxes = self._to_image_boxes(
                local_boxes=local_boxes,
                roi_bbox=roi_bbox,
                full_roi_shape=full_roi_shape,
                roi_index=roi_index,
                quadrant=quadrant,
                label=label,
            )

        golden_path, test_path, diff_path = save_roi_debug(
            position=artifact_scope,
            image_name=image_name,
            roi_index=roi_index,
            quadrant=quadrant,
            golden_roi=golden_roi,
            test_roi=test_roi,
        )

        return RoiResult(
            image_name=image_name,
            project_id=project_id,
            position=position,
            roi_index=roi_index,
            quadrant=quadrant,
            class_label=label,
            ssim=round(ssim_value, 4),
            edge_difference=round(edge_value, 4),
            mse=round(mse_value, 4),
            histogram_score=round(hist_value, 4),
            gradient_difference=round(grad_value, 4),
            local_difference=round(defect_ratio, 4),
            final_score=round(final_score, 4),
            prediction=prediction,
            defect_count=len(defect_boxes),
            defect_bounding_boxes=defect_boxes,
        )

    def _to_image_boxes(
        self,
        local_boxes: list[tuple[int, int, int, int]],
        roi_bbox: tuple[int, int, int, int],
        full_roi_shape,
        roi_index: int,
        quadrant: int,
        label: str,
    ) -> list[DefectBoundingBox]:
        roi_x, roi_y, _, _ = roi_bbox
        quadrant_x, quadrant_y = self._quadrant_offset(full_roi_shape, quadrant)
        boxes = []

        for x, y, width, height in local_boxes:
            x_min = roi_x + quadrant_x + x
            y_min = roi_y + quadrant_y + y
            boxes.append(
                DefectBoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_min + width,
                    y_max=y_min + height,
                    width=width,
                    height=height,
                    roi_index=roi_index,
                    quadrant=quadrant,
                    class_label=label,
                )
            )

        return boxes

    def _quadrant_offset(self, roi_shape, quadrant: int) -> tuple[int, int]:
        if not settings.enable_quadrants:
            return 0, 0

        height, width = roi_shape[:2]
        mid_h = height // 2
        mid_w = width // 2

        offsets = {
            1: (0, 0),
            2: (mid_w, 0),
            3: (0, mid_h),
            4: (mid_w, mid_h),
        }
        return offsets[quadrant]

