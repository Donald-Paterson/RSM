from config import *

from roi_extractor import yolo_to_bbox
from xml_parser import load_xml_annotations


def load_annotations(label_path, image_shape):

    if ANNOTATION_TYPE == "YOLO":

        return yolo_to_bbox(
            label_path,
            image_shape
        )

    elif ANNOTATION_TYPE == "XML":

        return load_xml_annotations(
            label_path
        )

    else:

        raise ValueError(
            f"Unsupported annotation type: {ANNOTATION_TYPE}"
        )