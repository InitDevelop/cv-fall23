import cv2
import mediapipe as mp

class DrawMesh:
    def __init__(self, static_image=False, max_n_face=1, confidence_detection=0.5, confidence_tracking=0.5):
        self.static_image = static_image
        self.max_n_face = max_n_face
        self.confidence_detection = confidence_detection
        self.confidence_tracking = confidence_tracking

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image, max_n_face,
                                                 refine_landmarks=True,
                                                 min_detection_confidence=confidence_detection,
                                                 min_tracking_confidence=confidence_tracking)
        self.left_eye_indices = list(range(474, 478))
        self.right_eye_indices = list(range(469, 473))
        self.last = None  # Initialize with None

    def face_mesh(self, frame):
        if self.last is not None:
            roi_x1 = max(0, int(self.last[1][0] / 4 - 200))
            roi_x2 = min(frame.shape[1] - 1, int(self.last[0][0] / 4 + 200))
            roi_y1 = max(0, int((self.last[0][1] + self.last[1][1]) / 8 - 50))
            roi_y2 = min(frame.shape[0] - 1, int((self.last[0][1] + self.last[1][1]) / 8 + 50))
            # roi_x1, roi_x2 = 0, 1920
            # roi_y1, roi_y2 = 0, 1080
        else:
            roi_x1, roi_x2 = 0, frame.shape[1]
            roi_y1, roi_y2 = 0, frame.shape[0]

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        frameRGB = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            self.last = [[0, 0], [0, 0]]
            for idx in self.left_eye_indices:
                point = results.multi_face_landmarks[0].landmark[idx]
                x, y = int(point.x * (roi_x2 - roi_x1) + roi_x1), int(point.y * (roi_y2 - roi_y1) + roi_y1)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                self.last[0][0] = self.last[0][0] + x
                self.last[0][1] = self.last[0][1] + y
            for idx in self.right_eye_indices:
                point = results.multi_face_landmarks[0].landmark[idx]
                x, y = int(point.x * (roi_x2 - roi_x1) + roi_x1), int(point.y * (roi_y2 - roi_y1) + roi_y1)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                self.last[1][0] = self.last[1][0] + x
                self.last[1][1] = self.last[1][1] + y
        else:
            # If no new landmarks, use the last known positions
            x, y = self.last[0]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            x, y = self.last[1]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            print("No new landmarks")
        # print((self.last[0][0] + self.last[1][0]) / 8, (self.last[0][1] + self.last[1][1]) / 8)

        return frame

    def get_eye_position(self, frame):
        if self.last is not None:
            roi_x1 = max(0, int(self.last[1][0] / 4 - 200))
            roi_x2 = min(frame.shape[1] - 1, int(self.last[0][0] / 4 + 200))
            roi_y1 = max(0, int((self.last[0][1] + self.last[1][1]) / 8 - 50))
            roi_y2 = min(frame.shape[0] - 1, int((self.last[0][1] + self.last[1][1]) / 8 + 50))
            # roi_x1, roi_x2 = 0, 1920
            # roi_y1, roi_y2 = 0, 1080
        else:
            roi_x1, roi_x2 = 0, 1919
            roi_y1, roi_y2 = 0, 1079

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        frameRGB = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            self.last = [[0, 0], [0, 0]]
            for idx in self.left_eye_indices:
                point = results.multi_face_landmarks[0].landmark[idx]
                x, y = int(point.x * (roi_x2 - roi_x1) + roi_x1), int(point.y * (roi_y2 - roi_y1) + roi_y1)
                self.last[0][0] = self.last[0][0] + x
                self.last[0][1] = self.last[0][1] + y
            for idx in self.right_eye_indices:
                point = results.multi_face_landmarks[0].landmark[idx]
                x, y = int(point.x * (roi_x2 - roi_x1) + roi_x1), int(point.y * (roi_y2 - roi_y1) + roi_y1)
                self.last[1][0] = self.last[1][0] + x
                self.last[1][1] = self.last[1][1] + y
        else:
            # If no new landmarks, use the last known positions
            x, y = self.last[0]
            x, y = self.last[1]
        return (self.last[0][0] + self.last[1][0]) / 8, (self.last[0][1] + self.last[1][1]) / 8

#
# if __name__ == "__main__":
#     mesh = DrawMesh()
#     capture_video(1280, 720, mesh.face_mesh)
