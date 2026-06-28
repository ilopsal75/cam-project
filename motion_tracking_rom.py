import cv2
import datetime

cap = cv2.VideoCapture(0)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = back_sub.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue
        detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    if detected:
        cv2.putText(frame, "MOTION DETECTED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        time_str = "ROMULUS // " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Romulus - Motion Feed", frame)

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()