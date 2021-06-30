import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib

Counter = 0

#EAR Hesabı
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

YAWN_THRESH = 20

#------------------------------------ Kafa Yönü Hesabı
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # Euler hesabı
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

#------------------------------------





faceCascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0) #VideoCapture open cv ile kameranın açılmasını sağlar. 0 yazmamızın sebebi bulunduğu cihazın kamerasını kullanmak istememizdir.
basla = time.time()



while True:
    # kare kare webcam den gelen görüntü yakalanıyor
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Tanımalanan yüzün etrafında yeşil bir kare oluşturuluyor
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    rects = detector(gray, 0)


    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #--------------
        distance = lip_distance(shape) #esneme tespiti
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1) # çizdirme


        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        ear = (leftEAR + rightEAR) / 2.0


        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)


        #----------------------------------------------------------------
        reprojectdst, euler_angle = get_head_pose(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        """for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))"""

        cv2.putText(frame, "Euler Acilari:", (20, 365), cv2.FONT_HERSHEY_SIMPLEX,  0.50, (255, 0, 0), thickness=2)
        cv2.putText(frame, "X: {:7.2f}".format(euler_angle[0, 0]), (20, 390), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Y: {:7.2f}".format(euler_angle[1, 0]), (20, 415), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Z: {:7.2f}".format(euler_angle[2, 0]), (20, 440), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=2)

        cv2.putText(frame, "Kafa Yonu:", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                    thickness=2)
        if euler_angle[0, 0] > 10:
            cv2.putText(frame, "Kafa Asagi Yone Bakiyor", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        thickness=2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)
        elif euler_angle[0, 0] < -10:
            cv2.putText(frame, "Kafa Yukari Yone Bakiyor", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        thickness=2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)
        elif euler_angle[1, 0] > 12:
            cv2.putText(frame, "Kafa Sag Yone Bakiyor", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        thickness=2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)
        elif euler_angle[1, 0] < -12:
            cv2.putText(frame, "Kafa Sol Yone Bakiyor", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        thickness=2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)
        else:
            cv2.putText(frame, "Karsiya Bakiyor", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        thickness=2)

        #------------------------------------------------------------------

        if ear < 0.27:
            simdiki = time.time()
            sonuc = simdiki - basla
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Goz kapalilik suresi: {:.1f}".format(sonuc)  , (250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)
        else:
            basla = time.time()
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Goz Acik", (250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if (distance > YAWN_THRESH):
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "ALARM !!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255),
                        thickness=2)


    # Sonuç ekranda gösteriliyor.
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# herşey tamamsa ekran yakalaması serbest bırakılıyor.
video_capture.release()
cv2.destroyAllWindows()
