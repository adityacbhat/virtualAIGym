import cv2
import mediapipe as mp
import math
import numpy as np
import os


##########################################################################################################################
################### TO BE EXECUTED ONLY AFTER TRAINING THE SECOND CNN CLASSIFICATION MODEL ###############################
from keras.models import model_from_json
json_file = open('dumbels.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
cnn.load_weights("dweights.h5")
print("Loaded model from disk")
##########################################################################################################################
##########################################################################################################################

mppose=mp.solutions.pose
mpdraw=mp.solutions.drawing_utils

pose=mppose.Pose()

cap=cv2.VideoCapture(0)
counter=1
updown=0
count=0
vidwriter = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25, (640,480))
while(True):
    ret,frame=cap.read()
    if(ret):
        framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(framergb)

        lmlist=[]
        if(results.pose_landmarks):
         #   mpdraw.draw_landmarks(frame,results.pose_landmarks,mppose.POSE_CONNECTIONS)

            for ids,lm in enumerate(results.pose_landmarks.landmark):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([ids,cx,cy])
                
            #    cv2.circle(frame,(cx,cy),5,(255,0,0),-1)
            
            angle=findAngle(lmlist,frame,12, 14, 16)
          #  angle2=findAngle(lmlist,frame,11, 13, 15)
            
            per = np.interp(angle, (220, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (350, 10))
            cv2.putText(frame, str(int(per)), (525,410),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0 ,255, 0), 3)
            
            
            cv2.rectangle(frame, (580, 10), (600, 350), (0,255,0), 2)
            cv2.rectangle(frame, (580, int(bar)), (600, 350), (0,0,255), cv2.FILLED)
            
            rx1=lmlist[22][1]-45
            ry1=lmlist[22][2]-45
            
            rx2=lmlist[22][1]+45
            ry2=lmlist[22][2]+45
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,255,0), 2)
            wrist=frame[ry1:ry2,rx1:rx2]
            result = cnn.predict(test_image)
            try:
                cv2.imwrite("img.jpg",wrist)
            
                d=withorwithout(wrist)
                if(d=="WITH DUMBBELLS"):
                    if(per==100 and updown==0):
                        count+=1
                        updown=1
                    if(per<100 and updown==1):
                        updown=0
                    cv2.putText(frame, "Counter: "+str(count), (30,150),
                               cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                    os.remove('img.jpg')
                    cv2.putText(frame, d, (30,100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
                if(d=="NO DUMBBELLS"):
                    cv2.putText(frame, "Counter: "+str(count), (30,150),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    cv2.putText(frame, d, (30,100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            except:
                continue




        try:
            cv2.imshow("frmae",frame)
        except:
            break
    k=cv2.waitKey(1)
    if(k==ord('q') or not ret):
        break
cap.release()    
cv2.destroyAllWindows()