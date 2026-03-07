import cv2
import pickle

video = cv2.VideoCapture('../data/videos/parking_1920_1080.mp4')
ret, frame = video.read()
video.release()

original_frame = frame.copy()
spots = []
drawing = False
start_x, start_y = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        spots.append((start_x, start_y, x - start_x, y - start_y))
        cv2.rectangle(frame, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow('Select Spots', frame)
        print('Spots selected: {}'.format(len(spots)))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if spots:
            spots.pop()
            frame[:] = original_frame.copy()
            for (x, y, w, h) in spots:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Select Spots', frame)
            print('Spot removed! Remaining: {}'.format(len(spots)))

cv2.imshow('Select Spots', frame)
cv2.setMouseCallback('Select Spots', draw_rectangle)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        pickle.dump(spots, open('../data/spots.p', 'wb'))
        print('Spots saved!')
        break

cv2.destroyAllWindows()