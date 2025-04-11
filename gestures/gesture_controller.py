import cv2
import mediapipe as mp

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Selfie Segmentation
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = hand.landmark

            # Tip landmarks
            index_tip = landmarks[8].y
            middle_tip = landmarks[12].y
            ring_tip = landmarks[16].y
            pinky_tip = landmarks[20].y

            fingers = [
                index_tip < landmarks[6].y,
                middle_tip < landmarks[10].y,
                ring_tip < landmarks[14].y,
                pinky_tip < landmarks[18].y
            ]

            if all(fingers):
                return "open_palm"
            elif fingers[0] and fingers[1] and not any(fingers[2:]):
                return "victory"
            elif fingers[0] and not any(fingers[1:]):
                return "index_up"

        return None

    def segment_body(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb)
        mask = results.segmentation_mask > 0.1

        bg_mask = (~mask).astype("uint8") * 255
        fg_mask = mask.astype("uint8") * 255

        return fg_mask
