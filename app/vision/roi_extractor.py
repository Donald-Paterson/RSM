from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def load_annotations(xml_path: str | Path) -> list[dict]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    components = []

    for obj in root.findall(".//points"):
        label = obj.attrib.get("label", "component")
        points_text = obj.attrib.get("points", "")
        points = []

        for point in points_text.split(";"):
            x, y = point.split(",")
            points.append([int(float(x)), int(float(y))])

        components.append(
            {
                "label": label,
                "points": np.array(points, dtype=np.int32),
            }
        )

    return components


def extract_polygon_rois(image: np.ndarray, components: list[dict]) -> list[dict]:
    rois = []

    for component in components:
        polygon = component["points"]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        masked = cv2.bitwise_and(image, image, mask=mask)
        x, y, width, height = cv2.boundingRect(polygon)

        rois.append(
            {
                "roi": masked[y : y + height, x : x + width],
                "label": component["label"],
                "polygon": polygon,
                "bbox": (x, y, x + width, y + height),
                "mask": mask[y : y + height, x : x + width],
            }
        )

    return rois


def divide_into_quadrants(roi: np.ndarray) -> list[np.ndarray]:
    height, width = roi.shape[:2]
    mid_h = height // 2
    mid_w = width // 2

    return [
        roi[0:mid_h, 0:mid_w],
        roi[0:mid_h, mid_w:width],
        roi[mid_h:height, 0:mid_w],
        roi[mid_h:height, mid_w:width],
    ]
