# Built-in function Face detection test
# this code is only for test
import sys
sys.path.append('C:\\GITHUB\\cv-fall23')
import mediapipe as mp
from cv_functions.capture_video import *


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

    def face_mesh(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) 
                            for idx in self.left_eye_indices]
                right_eye = [(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) 
                             for idx in self.right_eye_indices]

                for point in left_eye + right_eye:
                    x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        return frame

if __name__ == "__main__":
    mesh = DrawMesh()
    capture_video(1280, 720, mesh.face_mesh)
