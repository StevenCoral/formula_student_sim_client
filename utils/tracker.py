import numpy as np
import queue


class ObjectTracker:
    def __init__(self, initial_position):
        self.position = np.array(initial_position)
        self.detections = [np.array(initial_position)]

        self.active = False
        self.can_deactivate = False
        self.proximity_distance = 0.5
        self.activation_thresh = 5
        self.deactivation_thresh = 3

        self.max_detections = 10
        self.num_detections = 1
        self.num_misdetections = 0

    def process_detection(self, new_point):
        # if not self.check_proximity(new_point):
        #     if self.can_deactivate:
        #         self.num_misdetections += 1
        #         # More deactivation logic
        #     return False
        # else:
        detection = np.array(new_point)
        self.detections.append(detection)
        self.num_detections += 1
        if self.num_detections > self.max_detections:
            dump = self.detections.pop(0)
            self.num_detections -= 1
            self.position = (self.position * self.num_detections - dump) / (self.num_detections - 1)
        self.position = (self.position * (self.num_detections - 1) + detection) / self.num_detections
        if self.num_detections >= self.activation_thresh:
            self.active = True
        pass
        # return True

    def check_proximity(self, point):
        # Assuming point is a size-3 numpy array in the order of x-y-z.
        if np.linalg.norm(point - self.position) <= self.proximity_distance:
            return True
        else:
            return False


class TrackerManager:
    def __init__(self, point):
        pass


if __name__ == "__main__":
    tracker_test = ObjectTracker(np.array([0]))
    tracker_test.proximity_distance = 2.0
    for idx in range(12):
        tracker_test.process_detection(np.array([1]))
        if idx == 9:
            pass
        print(idx, tracker_test.position)

    for idx in range(11):
        tracker_test.process_detection(np.array([2]))
        print(idx, tracker_test.position)

    a=5
