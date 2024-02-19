from supervision import detection, Position
from ultralytics import YOLO
from playsound import playsound
import supervision as sv
import cv2
import threading



cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

# Флаг для управления основным циклом
running = True
# Глобальная переменная для хранения потока воспроизведения звука
sound_thread = None

def frame_generate():
    ret, frame = cap.read()
    return frame

def play_sound():
    playsound("alert.wav")


# Главная задача - обнаружить человека
def detect_person(frame, detections):
    global sound_thread
    for _, _, confidence, class_id, _, _ in detections:
        if model.names[class_id] == "person":
            print("Person detected with confidence:", confidence)
            if confidence > 0.7:
                if sound_thread is None or not sound_thread.is_alive():
                    sound_thread = threading.Thread(target=play_sound)
                    sound_thread.start()


# Ядро, которое обнаруживает и делает аннотацию
def detect(frame):
    result = model(frame)[0]

    detections = sv.Detections.from_ultralytics(result)
    detect_person(frame, detections)

    # labels можно изменить, отображает названия
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

def main_loop():
    global running
    while running:
        frame = frame_generate()
        frame = detect(frame)
        display_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False  # Если нажата клавиша q, выставляем флаг в False

if __name__ == "__main__":
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()

    # Ждем завершения основного потока
    main_thread.join()

    # Очищаем ресурсы
    cap.release()
    cv2.destroyAllWindows()