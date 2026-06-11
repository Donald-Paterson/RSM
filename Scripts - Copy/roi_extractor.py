import cv2
import numpy as np
import xml.etree.ElementTree as ET


# =========================================================
# LOAD XML ANNOTATIONS
# =========================================================

def load_annotations(xml_path):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    components = []

    point_objects = root.findall(
        ".//points"
    )

    print(
        f"TOTAL POINT OBJECTS FOUND: {len(point_objects)}"
    )

    for obj in point_objects:

        label = obj.attrib.get(

            "label",

            "component"
        )

        points_str = obj.attrib.get(

            "points",

            ""
        )

        points = []

        # =================================================
        # CONVERT STRING TO NUMPY POINTS
        # =================================================

        for p in points_str.split(";"):

            x, y = p.split(",")

            points.append([

                int(float(x)),
                int(float(y))

            ])

        polygon = np.array(

            points,

            dtype=np.int32
        )

        # =================================================
        # STORE COMPONENT
        # =================================================

        components.append({

            "label": label,

            "points": polygon
        })

        print(f"Loaded Component: {label}")

        print(f"Points Count: {len(points)}")

    print(f"TOTAL COMPONENTS: {len(components)}")

    return components


# =========================================================
# POLYGON ROI EXTRACTION
# =========================================================

def extract_polygon_rois(

    image,
    components
):

    rois = []

    for comp in components:

        polygon = comp["points"]

        label = comp["label"]

        # =================================================
        # CREATE EMPTY MASK
        # =================================================

        mask = np.zeros(

            image.shape[:2],

            dtype=np.uint8
        )

        # =================================================
        # DRAW FILLED POLYGON
        # =================================================

        cv2.fillPoly(

            mask,

            [polygon],

            255
        )

        # =================================================
        # APPLY MASK
        # =================================================

        masked = cv2.bitwise_and(

            image,
            image,

            mask=mask
        )

        # =================================================
        # BOUNDING RECTANGLE
        # =================================================

        x, y, w, h = cv2.boundingRect(
            polygon
        )

        cropped = masked[
            y:y+h,
            x:x+w
        ]

        # =================================================
        # REMOVE BLACK BORDER AREA
        # =================================================

        roi_mask = mask[
            y:y+h,
            x:x+w
        ]

        # =================================================
        # SAVE ROI
        # =================================================

        rois.append({

            "roi": cropped,

            "label": label,

            "polygon": polygon,

            "bbox": (

                x,
                y,
                x+w,
                y+h
            ),

            "mask": roi_mask
        })

    print(f"FINAL ROI COUNT: {len(rois)}")

    return rois


# =========================================================
# DIVIDE ROI INTO 4 QUADRANTS
# =========================================================

def divide_into_quadrants(roi):

    h, w = roi.shape[:2]

    mid_h = h // 2
    mid_w = w // 2

    q1 = roi[
        0:mid_h,
        0:mid_w
    ]

    q2 = roi[
        0:mid_h,
        mid_w:w
    ]

    q3 = roi[
        mid_h:h,
        0:mid_w
    ]

    q4 = roi[
        mid_h:h,
        mid_w:w
    ]

    return [

        q1,
        q2,
        q3,
        q4
    ]


# =========================================================
# DRAW POLYGONS ON IMAGE
# FOR VISUALIZATION / DEBUGGING
# =========================================================

def draw_polygons(

    image,
    components
):

    output = image.copy()

    for comp in components:

        polygon = comp["points"]

        label = comp["label"]

        cv2.polylines(

            output,

            [polygon],

            True,

            (0,255,0),

            2
        )

        x, y = polygon[0]

        cv2.putText(

            output,

            label,

            (x, y-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0,255,0),

            2
        )

    return output