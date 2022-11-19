import cv2
import mediapipe as mp

# 사용할 색 BGR 정의
red=(0,0,255)
green=(0,200,0)
white=(255,255,255)
black = (0,0,0)
yellow= (0,288,255)
pink=(166,97,243)

## 얼굴에 씌워줄 마스크 정의
image_mask1 = cv2.imread('python\\opencv\\project\\pink.png', cv2.IMREAD_UNCHANGED)
image_mask2 = cv2.imread('python\\opencv\\project\\yellow.png', cv2.IMREAD_UNCHANGED)
image_mask1=cv2.resize(image_mask1,(400,280))
image_mask2=cv2.resize(image_mask2,(400,280))

image_mask1_h, image_mask1_w, _ = image_mask1.shape
image_mask2_h, image_mask2_w, _ = image_mask2.shape

## 클릭할 마스크 아이콘 정의 (아이콘 클릭시 마스크 씌워줌)
image_icon1 = cv2.imread('python\\opencv\\project\\icons\\pink_icon.png')
image_icon2 = cv2.imread('python\\opencv\\project\\icons\\yellow_icon.png')
image_icon1 = cv2.resize(image_icon1, (120, 100))
image_icon2 = cv2.resize(image_icon2, (120, 100))
mp_face_detection = mp.solutions.face_detection   ## 얼굴 검출을 위한 face_detection 모듈 사용
mp_drawing = mp.solutions.drawing_utils  ## 얼굴의 특징을 그리기 위한 drawing_utils 모듈 사용

## 동영상 파일 열기
cap = cv2.VideoCapture('python\\opencv\\project\\twokids.mp4')
a = False
b = False
# 아이콘 클릭 마우스 이벤트 함수
def mouse_event(event, x, y, flags, param):
    global a, b
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < 110:
            b=False
            a = True
        elif y >= 120:
            a =False
            b =True
        print(x,y)

# 투명한 이미지 오버레이 함수
def overlay(img, x, y, w, h, overlay_img):  ## 이미지 띄울 화면, x 좌표, y 좌표 가로길이, 세로길이, 덮어씌울 이미지
    alpha = overlay_img[:,:,3] ## BGRA 에서 A만
    mask_img = alpha / 255  ## 0 ~ 1 사이 값 (1: 불투명)
    for c in range(0,3): # channel BGR
        img[y-h//2+30:y+h//2+30,x-w//2:x+w//2,c] = (overlay_img[:,:,c] * mask_img) + (img[y-h//2+30:y+h//2+30,x-w//2:x+w//2,c] * (1-mask_img))


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        try:
            success, image = cap.read()
            if not success:
                cap = cv2.VideoCapture('python\\opencv\\project\\twokids.mp4')
                success, image = cap.read()
                # 동영상 종료시 탈출
            # 영상 크기 조정
            resized_image = cv2.resize(image,(1280, 720))
            # 영상 좌우반전
            flip_image = cv2.flip(resized_image, 1)
            cv2.imshow('Choose Your Mask_Team2 Project', flip_image)
            flip_image.flags.writeable = False
            flip_image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)   ## mediapipe는 rgb // opencv는 bgr
            results = face_detection.process(flip_image)
            flip_image.flags.writeable = True
            flip_image = cv2.cvtColor(flip_image, cv2.COLOR_RGB2BGR)
            # 영상 가로, 세로 길이 정의
            h,w,_ = flip_image.shape
            if results.detections: #얼굴이 검출 됐을 때
                if a==True: # 분홍이를 누르면
                    for detection in results.detections:
                        keypoints=detection.location_data.relative_keypoints
                        nose=keypoints[2]
                        nose=(int(nose.x*w), int(nose.y*h))
                        overlay(flip_image,*nose,image_mask1_w, image_mask1_h, image_mask1)
                elif b == True: # 노랭이를 누르면
                    for detection in results.detections:
                        keypoints=detection.location_data.relative_keypoints
                        nose=keypoints[2]
                        nose=(int(nose.x*w), int(nose.y*h))
                        overlay(flip_image,*nose,image_mask2_w, image_mask2_h, image_mask2)
            flip_image[10:110, 10:130] = image_icon1
            flip_image[130:230, 10:130] = image_icon2
            cv2.imshow('Choose Your Mask_Team2 Project', flip_image)
            cv2.setMouseCallback('Choose Your Mask_Team2 Project',mouse_event)
            if cv2.waitKey(1) > 0:
                break
        except:
            pass
cap.release()
cv2.destroyAllWindows()
