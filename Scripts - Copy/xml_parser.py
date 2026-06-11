import xml.etree.ElementTree as ET


def load_xml_annotations(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    components = []

    # =====================================================
    # CVAT POINTS FORMAT
    # =====================================================

    points_objects = root.findall(".//points")

    print(f"TOTAL POINT OBJECTS FOUND: {len(points_objects)}")

    for idx, obj in enumerate(points_objects):

        try:

            label = obj.get(
                "label",
                f"component_{idx}"
            )

            points_str = obj.get("points")

            if points_str is None:

                continue

            polygon_points = []

            pairs = points_str.split(";")

            for pair in pairs:

                x, y = pair.split(",")

                polygon_points.append(

                    (
                        int(float(x)),
                        int(float(y))
                    )
                )

            components.append({

                "label": label,

                "type": "polygon",

                "points": polygon_points
            })

            print(f"Loaded Component: {label}")

            print(f"Points Count: {len(polygon_points)}")

        except Exception as e:

            print(f"XML Parse Error: {e}")

    print(f"TOTAL COMPONENTS: {len(components)}")

    return components