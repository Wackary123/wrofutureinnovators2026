import cv2
import importlib
import time

HISTORY_WINDOW = 12
MIN_HISTORY_MATCHES = 4
POSITION_STABILITY_TOLERANCE = 37
SIZE_STABILITY_TOLERANCE = 12
SIZE_CHANGE_DELTA_TOLERANCE = 14
SIZE_CHANGE_FREQUENCY_THRESHOLD = 0.75
KEEP_UNCONFIRMED_DETECTIONS = False # no touchies
MATCH_SCORE_TOLERANCE = 70
NESTED_BOX_MARGIN = 3 # no touchies
NESTED_REQUIRE_SAME_COLOR = True # no touchies
MIN_CONTOUR_AREA = 450 # no touchies
CAMERA_SIZE = (640, 480)
CAMERA_FORMAT = "BGR888"
CAMERA_WARMUP_SECONDS = 0.2

RED_LOWER_1 = (0, 120, 80)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (170, 120, 80)
RED_UPPER_2 = (179, 255, 255)
BLUE_LOWER = (95, 100, 70)
BLUE_UPPER = (130, 255, 255)
GREEN_LOWER = (40, 40, 40)
GREEN_UPPER = (70, 255, 255)


def create_picamera2_instance():
    """Loads Picamera2 only at runtime so this file can still be edited off-device."""
    picamera2_module = importlib.import_module("picamera2")
    return picamera2_module.Picamera2()

"""
Detects objects based on color,  from computer camera for now
"""
def detect_color(frame):
    """
    Detects color patches in the image
    param frame: the image frame from the camera
    return: list of tuples,  each tuple is a patch of color  of form (color, position, size)
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV (OpenCV hue range is 0..179).
    color_ranges = {
        "red": [(RED_LOWER_1, RED_UPPER_1), (RED_LOWER_2, RED_UPPER_2)],
        "blue": [(BLUE_LOWER, BLUE_UPPER)],
        "green": [(GREEN_LOWER, GREEN_UPPER)]
    }

    detected_colors = []
    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                detected_colors.append((color_name, (x + w // 2, y + h // 2), (w, h)))

    return detected_colors

def filter_detections(past_detections):
    """
    Filters detections to keep only stable/static ones.
    Detections that move or change size frequently over time are removed.
    param past_detections: list of frame detections, each frame detection is a
    list of tuples of form (color, position, size)
    return: filtered detections for the latest frame
    """
    if not past_detections:
        return []

    # Work on a recent history window to avoid stale detections influencing the result.
    history = past_detections[-HISTORY_WINDOW:]
    latest_detections = history[-1]

    filtered = []
    for color_name, (x, y), (w, h) in latest_detections:
        matched_history = []

        # Collect same-color detections from each frame that are close to the current one.
        for frame_detections in history:
            best_match = None
            best_score = None

            for c_name, (fx, fy), (fw, fh) in frame_detections:
                if c_name != color_name:
                    continue

                score = abs(fx - x) + abs(fy - y) + abs(fw - w) + abs(fh - h)
                if best_score is None or score < best_score:
                    best_score = score
                    best_match = (fx, fy, fw, fh)

            if best_match is not None and best_score <= MATCH_SCORE_TOLERANCE:
                matched_history.append(best_match)

        # Without enough history we cannot confirm stability yet.
        if len(matched_history) < MIN_HISTORY_MATCHES:
            if KEEP_UNCONFIRMED_DETECTIONS:
                filtered.append((color_name, (x, y), (w, h)))
            continue

        xs = [item[0] for item in matched_history]
        ys = [item[1] for item in matched_history]
        ws = [item[2] for item in matched_history]
        hs = [item[3] for item in matched_history]

        is_position_static = (max(xs) - min(xs) <= POSITION_STABILITY_TOLERANCE) and (max(ys) - min(ys) <= POSITION_STABILITY_TOLERANCE)
        is_size_static = (max(ws) - min(ws) <= SIZE_STABILITY_TOLERANCE) and (max(hs) - min(hs) <= SIZE_STABILITY_TOLERANCE)

        # Remove detections whose size changes too frequently over time.
        size_change_count = 0
        for i in range(1, len(matched_history)):
            prev_w, prev_h = matched_history[i - 1][2], matched_history[i - 1][3]
            curr_w, curr_h = matched_history[i][2], matched_history[i][3]
            if abs(curr_w - prev_w) > SIZE_CHANGE_DELTA_TOLERANCE or abs(curr_h - prev_h) > SIZE_CHANGE_DELTA_TOLERANCE:
                size_change_count += 1

        change_frequency = size_change_count / max(1, len(matched_history) - 1)
        if change_frequency >= SIZE_CHANGE_FREQUENCY_THRESHOLD:
            continue

        # Keep only detections that are stable in both position and size.
        if not (is_position_static and is_size_static):
            continue

        filtered.append((color_name, (x, y), (w, h)))

    return remove_nested_detections(filtered)


def _to_corners(position, size):
    """Converts (center_x, center_y), (w, h) to (left, top, right, bottom)."""
    x, y = position
    w, h = size
    left = x - w // 2
    top = y - h // 2
    right = x + w // 2
    bottom = y + h // 2
    return left, top, right, bottom


def _is_inside(inner_detection, outer_detection, margin):
    """Returns True when inner_detection box is inside outer_detection box."""
    _, inner_pos, inner_size = inner_detection
    _, outer_pos, outer_size = outer_detection
    il, it, ir, ib = _to_corners(inner_pos, inner_size)
    ol, ot, or_, ob = _to_corners(outer_pos, outer_size)
    return il >= ol + margin and it >= ot + margin and ir <= or_ - margin and ib <= ob - margin


def remove_nested_detections(detections):
    """Removes detections whose bounding boxes are fully nested in another one."""
    keep = [True] * len(detections)

    for i, det_i in enumerate(detections):
        color_i, _, size_i = det_i
        area_i = size_i[0] * size_i[1]

        for j, det_j in enumerate(detections):
            if i == j:
                continue

            color_j, _, size_j = det_j
            if NESTED_REQUIRE_SAME_COLOR and color_i != color_j:
                continue

            area_j = size_j[0] * size_j[1]
            if _is_inside(det_i, det_j, NESTED_BOX_MARGIN):
                # Keep one box if they are effectively duplicates.
                if area_i < area_j or (area_i == area_j and i > j):
                    keep[i] = False
                    break

    return [det for idx, det in enumerate(detections) if keep[idx]]

if __name__ == "__main__":
    try:
        picam2 = create_picamera2_instance()
    except ModuleNotFoundError as exc:
        raise SystemExit("Picamera2 is required. Install python3-picamera2 on Raspberry Pi.") from exc

    camera_config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": CAMERA_FORMAT}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(CAMERA_WARMUP_SECONDS)

    detection_results = []
    draw_colors = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
    }

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detected_colors = detect_color(frame)
            detection_results.append(detected_colors)
            if len(detection_results) > HISTORY_WINDOW:
                detection_results.pop(0)

            relevant_detections = filter_detections(detection_results)

            for color_name, position, size in relevant_detections:
                x, y = position
                w, h = size
                draw_color = draw_colors.get(color_name, (255, 255, 255))
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), draw_color, 2)
                cv2.putText(frame, color_name, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

            cv2.imshow("Color Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
