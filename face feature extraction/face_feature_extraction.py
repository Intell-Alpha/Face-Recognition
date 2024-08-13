import mediapipe as mp
import cv2 as cv
from FaceMeshDetector import FaceMeshDetector

# some feature names
# 1.	silhouette
# 2.	lipsUpperOuter
# 3.	leftEyeIris
# 4.	rightEyeIris

right_eye=[246, 161, 160, 159, 158, 157, 173,7, 163, 144, 145, 153, 154, 155, 133, 30, 29, 27, 28, 56, 190, 25, 110, 24, 23, 22, 26, 112, 243, 225, 224, 223, 222, 221, 189, 31, 228, 229, 230, 231, 232, 233, 244, 111, 117, 118, 119, 120, 121, 128, 245]
left_eye=[466, 388, 387, 386, 385, 384, 398,263, 249, 390, 373, 374, 380, 381, 382, 362,467, 260, 259, 257, 258, 286, 414,359, 255, 339, 254, 253, 252, 256, 341, 463,342, 45, 444, 443, 442, 441, 413,446, 261, 448, 449, 450, 451, 452, 453, 464,372, 340, 346, 347, 348, 349, 350, 357, 465]


def main():
    img=cv.imread("prabhas.jpeg")
    
    detector = FaceMeshDetector()
    
    # feature="lipsUpperOuter"
    # feature_index = detector.keypoints[feature]

    feature="right_eye"
    feature_index=right_eye

    # feature="left_eye"
    # feature_index=left_eye





    extra=0.05
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
        # cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,255,0),2)

        width=end_x-start_x
        height=end_y-start_y

        start_x-=int(width*extra)
        start_y-= int(height*extra)
        end_x+= int(width*extra)
        end_y+= int(height*extra)

        

        sub_image=img[start_y:end_y,start_x:end_x]
        sub_image=cv.resize(sub_image,[sub_image.shape[1]*3,sub_image.shape[0]*3])
        cv.imshow(f"extracted {feature}",sub_image)

     
        cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,255,0),2)
        cv.imshow(f"detected {feature}", img)
    

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


