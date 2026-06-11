def split_into_quadrants(roi):

    if roi is None:

        return []

    if roi.size == 0:

        return []

    h, w = roi.shape[:2]

    if h < 4 or w < 4:

        return [roi]

    mid_x = w // 2
    mid_y = h // 2

    q1 = roi[0:mid_y, 0:mid_x]

    q2 = roi[0:mid_y, mid_x:w]

    q3 = roi[mid_y:h, 0:mid_x]

    q4 = roi[mid_y:h, mid_x:w]

    quadrants = []

    for q in [q1, q2, q3, q4]:

        if q is None:
            continue

        if q.size == 0:
            continue

        quadrants.append(q)

    return quadrants