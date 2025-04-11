import json

class StoryController:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.story = json.load(f)
        self.scenes = self.story["scenes"]
        self.current_scene_idx = 0

    def get_current_scene(self):
        return self.scenes[self.current_scene_idx]

    def next_scene(self):
        if self.current_scene_idx + 1 < len(self.scenes):
            self.current_scene_idx += 1
            return True
        return False

    def previous_scene(self):
        if self.current_scene_idx - 1 >= 0:
            self.current_scene_idx -= 1
            return True
        return False

    def reset_story(self):
        self.current_scene_idx = 0

    def skip_scene(self):
        self.current_scene_idx = min(self.current_scene_idx + 2, len(self.scenes) - 1)

    def get_expected_gesture(self):
        return self.get_current_scene()["gesture"]
