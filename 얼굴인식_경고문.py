import cv2
import mediapipe as mp

red=(0,0,255)
green=(0,200,0)
white=(255,255,255)
black = (0,0,0)
shade=(1010+1,80+1)

## 얼굴 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection   ## 얼굴 검출을 위한 face_detection 모듈 사용
mp_drawing = mp.solutions.drawing_utils  ## 얼굴의 특징을 그리기 위한 drawing_utils 모듈 사용

## 동영상 파일 열기
cap = cv2.VideoCapture('python\\opencv\\project\\man.mp4')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.88) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
              # 동영상 종료시 탈출
           
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   ## mediapipe는 rgb , opencv는 bgr
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h,w,_=image.shape #높이, 넓이 , channel (필요 없을 시 _)

        ## 검출된 얼굴이 있다면 
        if results.detections:
            cv2.rectangle(image,(1000,20),(1270,100), black,4)
            cv2.putText(image, 'MASK', (1010, 80), cv2.FONT_HERSHEY_TRIPLEX, 2, black)
            cv2.putText(image, 'MASK', shade, cv2.FONT_HERSHEY_TRIPLEX, 2, black)
            cv2.line(image,(1205,35),(1255,85),red,10)
            cv2.line(image,(1255,35),(1205,85),red,10)
                
        else:
            cv2.rectangle(image,(1000,20),(1270,100), black,4)
            cv2.putText(image, 'MASK', (1010, 80), cv2.FONT_HERSHEY_TRIPLEX, 2, black)
            cv2.putText(image, 'MASK', shade, cv2.FONT_HERSHEY_TRIPLEX, 2, black)
            cv2.circle(image, (1230,60), 25, green, 10, cv2.LINE_AA)
            
    

        cv2.imshow('Mask Detection_Team2 Project',cv2.resize(image,None,fx=0.5,fy=0.5))       

        if cv2.waitKey(1) == 27:  # esc키 누를 시, 동영상 종료
            break

cap.release()
cv2.destroyAllWindows()