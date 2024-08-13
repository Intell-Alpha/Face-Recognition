import mediapipe as mp
import cv2 as cv
from FaceMeshDetector import FaceMeshDetector



def main():
    img=cv.imread("prabhas.jpeg")
    
    detector = FaceMeshDetector()
    
    feature="silhouette"
    feature_index = detector.keypoints[feature]

    extra=0.1
    img,faces = detector.findFaceMesh(img,False)
    if len(faces)!=0:
        pts_x=[]
        pts_y=[]
        for i in feature_index :
            pts_x.append(faces[0][i][0])
            pts_y.append(faces[0][i][1])
        # coordinates of atop-left to bottom-right
        start_x=min(pts_x)
        start_y=min(pts_y) 
        end_x=max(pts_x) 
        end_y=max(pts_y)
        print("old -> ",start_x,end_x,start_y,end_y)
        cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,255,0),2)

        width=end_x-start_x
        height=end_y-start_y
        print(width,height)

        start_x-= int(width*extra)
        start_y-= int(height*extra)
        end_x+= int(width*extra)
        end_y+= int(height*extra)
        print("new -> ",start_x,end_x,start_y,end_y)
        cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,0,255),2)
     
        
        cv.imshow("image", img)
    

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


