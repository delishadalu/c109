import cv2
import mediapipe as mp ## helps to detect human postures and hand gestures

cam=cv2.VideoCapture(0)

mp_hands = mp.solutions.hands ## detect the hand all all the 21 joints
mp_drawing = mp.solutions.drawing_utils ## helps to draw a line connecting all the joints


tiptop = [4, 8, 12, 16, 20]

hands= mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5)


def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks: # checking if we recieved handlandmarks
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[handNo].landmark # values x, y ,z
        
        print(landmarks)
        
        fingers=0
        
        for index in tiptop:
            
            fingertip=landmarks[index].y
            fingerbottom=landmarks[index-2].y
            
            thumbtip=landmarks[index].x
            thumbbottom=landmarks[index-2].x
            
            if index==4:
                if thumbtip>thumbbottom:
                    fingers+=1
                    
            else:
                if fingertip<fingerbottom:
                    fingers+=1
                    
                    
        text=f"no.of fingers opened={fingers}"            
        cv2.putText(image, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                
            
            
        
        
        

while True:
    ret,image=cam.read()

    image=cv2.flip(image,1) # 1> flip horizonbtally
    
    results = hands.process(image) # detect the hand in each image, hand()--> 2 arguments

    # draw the landmarks in the hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
    countFingers(image, results.multi_hand_landmarks)
    
    cv2.imshow("image",image)

    if cv2.waitKey(1) == 32:
        break

cam.release()
cv2.destroyAllWindows()