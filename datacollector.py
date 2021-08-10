import cv2
import mediapipe as mp
import math
import numpy as np
import os





cap=cv2.VideoCapture(0)
counter=1
while(True):
    ret,frame=cap.read()
    if(ret):
        framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(framergb)

        lmlist=[]
        if(results.pose_landmarks):      

            for ids,lm in enumerate(results.pose_landmarks.landmark):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([ids,cx,cy])
            
            angle=findAngle(lmlist,frame,12, 14, 16)            
            per = np.interp(angle, (190, 330), (0, 100))
            bar = np.interp(angle, (190, 330), (350, 10))
            cv2.putText(frame, "Right: "+str(int(per)), (30,50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            
            cv2.rectangle(frame, (580, 10), (600, 350), (0,255,0), 2)
            cv2.rectangle(frame, (580, int(bar)), (600, 350), (0,0,255), cv2.FILLED)
            
            rx1=lmlist[22][1]-45
            ry1=lmlist[22][2]-45
            
            rx2=lmlist[22][1]+45
            ry2=lmlist[22][2]+45
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,255,0), 2)
            wrist=frame[ry1:ry2,rx1:rx2]

           
            try:
                cv2.imwrite("with"+str(counter)+".jpg",wrist)
                counter+=1
            except:
                pass
        
        try:
            cv2.imshow("frmae",frame)
        except:
            break
    k=cv2.waitKey(1)
    if(k==ord('q') or not ret):
        break
cap.release()    
cv2.destroyAllWindows()