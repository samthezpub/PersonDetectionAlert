from supervision import detection, Position
from ultralytics import YOLO
import supervision as sv
import cv2

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

def frame_generate():
    ret, frame = cap.read()
    return frame


def detect_person(frame, detections):
    person_found = False
    for _, _, confidence, class_id, _, _ in detections:
        if model.names[class_id] == "person":
            print("Person detected with confidence:", confidence)
            if confidence > 0.7:
                person_found = True
            break
    return person_found

def detect(frame):
    result = model(frame)[0]

    detections = sv.Detections.from_ultralytics(result)
    detect_person(frame, detections)

    labels = [
        f"{model.names[class_id]}"
        for xyxy, mask, confidence, class_id, tracker_id, data in detections
    ]

    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame

def display_frame(frame):
    cv2.imshow('Video', frame)


if __name__ == "__main__":
    while(True):
        frame = frame_generate()
        frame = detect(frame)

        display_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()