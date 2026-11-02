
import cv2
import mediapipe as mp
import os


def detect_peace_sign(landmarks):
    # Memeriksa posisi jari
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y

    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    # Memastikan hanya telunjuk dan tengah yang terangkat
    return (
        index_up
        and middle_up
        and ring_down
        and pinky_down
    )


# Memeriksa lokasi model
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model hand_landmarker.task tidak ditemukan!")

# Inisialisasi MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Membuat koneksi antar landmark
connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# Membuka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak dapat dibuka!")

# Membuat hand detector
with HandLandmarker.create_from_options(options) as detector:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal mengambil frame!")
            break

        # Mirror kamera
        frame = cv2.flip(frame, 1)

        # Konversi BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Membuat objek MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp video
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        timestamp_ms = int(timestamp * 1000)

        # Deteksi tangan
        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Menggambar landmark
        # Status gesture setiap frame
        gesture_detected = False

        # Menggambar landmark
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:

                height, width, _ = frame.shape

            if detect_peace_sign(hand_landmarks):
                gesture_detected = True

                cv2.putText(
                        frame,
                        "PEACE SIGN DETECTED",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                cv2.putText(
                    frame,
                    "NORMAL GESTURE",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # Menggambar garis antar titik
            for connection in connections:
                start = connection.start
                end = connection.end

                x1 = int(hand_landmarks[start].x * width)
                y1 = int(hand_landmarks[start].y * height)

                x2 = int(hand_landmarks[end].x * width)
                y2 = int(hand_landmarks[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Menggambar titik landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "WHO ME",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Efek blur pada seluruh tampilan kamera
        if gesture_detected:
            frame = cv2.GaussianBlur(
                frame,
                (51, 51),
                0
            )
        # Menampilkan webcam
        cv2.imshow("Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Membersihkan kamera
cap.release()
cv2.destroyAllWindows()
