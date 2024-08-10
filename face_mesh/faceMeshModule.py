import mediapipe as mp
import cv2 as cv
import time

class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.static_image_mode, 
                                                  max_num_faces=self.max_num_faces,
                                                  min_detection_confidence=self.min_detection_confidence,
                                                  min_tracking_confidence=self.min_tracking_confidence)
        
        self.FACE_CONNECTIONS = self.mpFaceMesh.FACEMESH_CONTOURS
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1,color=[0,255,0])

    def findFaceMesh(self, img, draw=True):
        rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgbImg)
        faces=[]
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face=[]
                for lm in faceLms.landmark:
                    # print(lm)
                    ih,iw,ic=img.shape
                    x,y =int(lm.x+iw),int(lm.y+ih)
                    face.append([x,y])
                faces.append(face)
        return img,faces

def main():
    video_path = "resources/4.mp4"
    video = cv.VideoCapture(video_path)
    # video = cv.VideoCapture(0)
    prev_frame_time = 0
    new_frame_time = 0
    detector = FaceMeshDetector()

    while True:
        ret, img = video.read()
        if not ret: 
            break
        img,faces = detector.findFaceMesh(img)
        # if len(faces)!=0:
           # print(len(faces))
        new_frame_time = time.time() 
        fps = 1 / (new_frame_time - prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps)
        fps_text = "FPS: " + str(fps)

        cv.putText(img, fps_text, (28, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("face mesh", img)
        if cv.waitKey(10) & 0xFF == ord('q'): 
            break
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


