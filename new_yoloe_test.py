import cv2
from picamera2 import Picamera2
from ultralytics import YOLOE

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOE text-promptable model (NOT the -pf model)
model = YOLOE("yoloe-11s-seg.pt")

# Only detect these three prompts
model.set_classes(["dog", "phone", "clock", "hoodie", "computer", "box", "plant", "tape", "mona lisa", "vase", "hair", "person","table","light","fruit"])

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Run YOLOE model on the captured frame
    results = model.predict(frame)

    # Output the visual detection data
    annotated_frame = results[0].plot(boxes=True, masks=False)

    # Get inference time
    inference_time = results[0].speed["inference"]
    fps = 1000 / inference_time
    text = f"FPS: {fps:.1f}"

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()