import cv2
import time
import numpy as np

from stories.story_controller import StoryController
from gestures.gesture_controller import GestureController

gesture_cooldown = 1.5  # seconds
last_gesture_time = 0

def overlay_image(bg, fg, x=0, y=0):
    """Overlay an image with alpha channel onto a background."""
    if fg is None or fg.shape[2] != 4:
        return bg
    fg_h, fg_w = fg.shape[:2]
    if y + fg_h > bg.shape[0] or x + fg_w > bg.shape[1]:
        return bg  # Don't draw out of bounds

    alpha_s = fg[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        bg[y:y+fg_h, x:x+fg_w, c] = (
            alpha_s * fg[:, :, c] + alpha_l * bg[y:y+fg_h, x:x+fg_w, c]
        )
    return bg

def blend_background(bg, frame, mask):
    """Blend only the segmented body into the background scene."""
    frame = cv2.resize(frame, (bg.shape[1], bg.shape[0]))
    mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]))

    mask_3ch = cv2.merge([mask] * 3)
    inv_mask = cv2.bitwise_not(mask_3ch)

    fg_person = cv2.bitwise_and(frame, mask_3ch)
    bg_only = cv2.bitwise_and(bg, inv_mask)

    return cv2.add(fg_person, bg_only)

def draw_wrapped_text(image, text, x, y, max_width, font, scale, color, thickness, line_height):
    words = text.split(' ')
    line = ''
    for word in words:
        test_line = line + word + ' '
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w > max_width:
            cv2.putText(image, line, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
            y += line_height
            line = word + ' '
        else:
            line = test_line
    if line:
        cv2.putText(image, line, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def main():
    global last_gesture_time
    story = StoryController("stories/story_config.json")
    gesture = GestureController()
    cap = cv2.VideoCapture(0)

    instructions_shown = False
    instruction_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        scene = story.get_current_scene()
        bg = cv2.imread(scene['background'])
        if bg is None:
            print(f"[Warning] Background image not found: {scene['background']}")
            continue

        # Resize background to window size if needed
        screen_width = 1280
        scale = screen_width / bg.shape[1]
        bg = cv2.resize(bg, (screen_width, int(bg.shape[0] * scale)))

        # Segment and blend
        mask = gesture.segment_body(frame)
        final_frame = blend_background(bg, frame, mask)

        # Overlay
        if scene.get('overlay'):
            overlay = cv2.imread(scene['overlay'], cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                overlay = cv2.resize(overlay, (200, 200))
                final_frame = overlay_image(final_frame, overlay, x=30, y=30)

        # Add scene text (wrapped)
        draw_wrapped_text(final_frame, scene['text'], 30, final_frame.shape[0] - 110,
                          max_width=final_frame.shape[1] - 60, font=cv2.FONT_HERSHEY_SIMPLEX,
                          scale=0.8, color=(255, 255, 255), thickness=2, line_height=30)

        # Show gesture instructions for first 5 seconds
        if not instructions_shown:
            cv2.putText(final_frame, "✋ Open Palm = Next | ☝️ Index Up = Back | ✌️ Victory = Skip",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if time.time() - instruction_start > 5:
                instructions_shown = True

        # Detect gesture
        gesture_name = gesture.detect_gesture(frame)
        now = time.time()
        if gesture_name and (now - last_gesture_time > gesture_cooldown):
            last_gesture_time = now
            if gesture_name == "open_palm":
                story.next_scene()
            elif gesture_name == "victory":
                story.skip_scene()
            elif gesture_name == "index_up":
                story.previous_scene()

        # Keyboard fallback
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            story.next_scene()
        elif key == ord('s'):
            story.skip_scene()
        elif key == ord('b'):
            story.previous_scene()
        elif key == ord('r'):
            story.reset_story()

        # Show
        cv2.imshow("AR Story", final_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
