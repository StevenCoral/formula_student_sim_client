import numpy as np


class ObjectTracker:
    def __init__(self, initial_position, distance=1.0):
        self.position = np.array(initial_position)
        self.detections = [np.array(initial_position)]

        self.active = False
        self.can_deactivate = False
        self.proximity_distance = distance
        self.activation_thresh = 3
        self.deactivation_thresh = 3

        self.max_detections = 100
        self.num_detections = 1
        self.num_misdetections = 0

    def process_detection(self, new_point):
        detection = np.array(new_point)
        self.detections.append(detection)
        self.num_detections += 1
        if self.num_detections > self.max_detections:
            # Remove oldest detection from the list and its contribution to the moving average:
            dump = self.detections.pop(0)
            self.num_detections -= 1
            self.position = (self.position * self.num_detections - dump) / (self.num_detections - 1)
        # Add the new detection's values into the moving average:
        self.position = (self.position * (self.num_detections - 1) + detection) / self.num_detections
        if self.num_detections >= self.activation_thresh:
            self.active = True
        pass

    def check_proximity(self, point):
        # Assuming point is a size-3 numpy array in the order of x-y-z.
        if np.linalg.norm(point - self.position) <= self.proximity_distance:
            return True
        else:
            return False


class ConeTracker(ObjectTracker):
    COLOR_UNKNOWN = 0
    COLOR_BLUE = 1
    COLOR_YELLOW = 2

    # Values measured in a separate procedure:
    HUE_MIN_BLUE = 107
    HUE_MAX_BLUE = 111
    SAT_MIN_BLUE = 200
    SAT_MAX_BLUE = 255

    HUE_MIN_YELLOW = 20
    HUE_MAX_YELLOW = 40
    SAT_MIN_YELLOW = 80
    SAT_MAX_YELLOW = 160

    def __init__(self, initial_position):
        super().__init__(initial_position)
        self.color = ConeTracker.COLOR_UNKNOWN
        self.color_history = [ConeTracker.COLOR_UNKNOWN]

    def determine_color(self, hsv_image):
        resulting_color = estimate_cone_color(hsv_image)
        if resulting_color != self.COLOR_UNKNOWN:
            # Only change colors if a valid result is determined.
            self.color = resulting_color
        else:
            pass

        self.color_history.append(resulting_color)
        return resulting_color

    @classmethod
    def generate_histogram(cls, hsv_image):
        # Create a mask of known ranges of saturation (to differentiate cone from asphalt background):
        saturation_mask = np.logical_or(np.logical_and(hsv_image[:, :, 1] >= cls.SAT_MIN_YELLOW,
                                                       hsv_image[:, :, 1] <= cls.SAT_MAX_YELLOW),
                                        np.logical_and(hsv_image[:, :, 1] >= cls.SAT_MIN_BLUE,
                                                       hsv_image[:, :, 1] <= cls.SAT_MAX_BLUE))
        # Use masked cells to create a Hue histogram so we can differentiate between cone colors:
        histogram = np.bincount(hsv_image[saturation_mask, 0].flatten(), minlength=180)  # Faster than np.histogram()
        return histogram


def compare_trackers(centroid_pos, tracker_list):
    centroid_exists = False
    # Compare them against all the known tracked objects:
    for centroid in tracker_list:
        if centroid.check_proximity(centroid_pos):
            centroid_exists = True
            centroid.process_detection(centroid_pos)

    # If it is not close enough to an existing point, create a new tracker instance:
    if not centroid_exists:
        tracker_list.append(ConeTracker(centroid_pos))


# For use outside of the class:
def estimate_cone_color(hsv_image):
    histogram = ConeTracker.generate_histogram(hsv_image)
    # Summing histogram counts of blue and yellow:
    blue_score = np.sum(histogram[ConeTracker.HUE_MIN_BLUE:ConeTracker.HUE_MAX_BLUE])
    yellow_score = np.sum(histogram[ConeTracker.HUE_MIN_YELLOW:ConeTracker.HUE_MAX_YELLOW])
    if blue_score > yellow_score:
        return ConeTracker.COLOR_BLUE
    elif blue_score < yellow_score:
        return ConeTracker.COLOR_YELLOW
    else:
        return ConeTracker.COLOR_UNKNOWN


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
