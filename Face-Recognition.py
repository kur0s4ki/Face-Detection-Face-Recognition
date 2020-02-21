import face_recognition
import cv2
import numpy as np
import platform
import pickle
from fstring import fstring
import uuid
import math

global Just_Enrolled
known_face_encodings = []
known_face_metadata = []
ID_BD = []

# Parameters
Tolerance = 0.6  # choose a value between 0 and 1 , The lower tolerance is ,the strict face recognition becomes.
model = 'hog'  # Face Detection Method , Choose between hog/cnn , cnn requires a powerful machine.
Distance_Threshhold = 60  # Distance (CM) at which faces would be detected and recognized.
Distance_Method = 'Algebra'  # Choose between Algebra/Estimation , Algebra method is more accurate.
number_jitters = 1  # A Higher Number will slow the recognition process.
ID_Method = 'Counter'  # Choose Between Counter/UUID-Generator.
Auto_Enrollment = 'Enabled'  # Choose Between Enabled/Disabled.


def backup():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("[+] BACKING UP DATA ...")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("[+] LOADING DATA ...")
    except FileNotFoundError as e:
        print('{}'.format(e))
        print("[+] NO DATA FOUND , CREATING A NEW LIST\n")


def running_on_jetson_nano():
    return platform.machine() == "aarch64"


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720,
                                framerate=60, flip_method=0):
    return (
            fstring('nvarguscamerasrc ! video/x-raw(memory:NVMM), ') +
            fstring('width=(int){capture_width}, height=(int){capture_height}, ') +
            fstring('format=(string)NV12, framerate=(fraction){framerate}/1 ! ') +
            fstring('nvvidconv flip-method={flip_method} ! ') +
            fstring('video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ') +
            fstring('videoconvert ! video/x-raw, format=(string)BGR ! appsink')
    )


def get_ID():
    global ID_BD
    while True:
        ID = str(uuid.uuid4())
        # random = random.upper()
        ID = ID.replace("-", "")
        ID = ID[0:4]
        if not (ID in ID_BD):
            ID_BD.append(ID)
            break
    return ID


def ID_counter():
    global ID_BD
    ID = 0
    while True:
        if ID in ID_BD:
            ID += 1
        else:
            ID_BD.append(ID)
            break
    return ID


def register(face_encoding, face_image):
    face_image = cv2.resize(face_image, (75, 75))
    known_face_encodings.append(face_encoding)
    if ID_Method == 'UUID_Generator':
        known_face_metadata.append({
            "id": get_ID(),
            "face_image": face_image
        })
    else:
        known_face_metadata.append({
            "id": ID_counter(),
            "face_image": face_image
        })

    return known_face_metadata[0]['id']


def find_faces(face_encoding):
    metadata = None
    if len(known_face_encodings) == 0:
        return metadata
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index] < Tolerance:
        metadata = known_face_metadata[best_match_index]
    return metadata


def DummyFuncEnrolled(key):
    print("[+] KNOWN USER. \n[+] AFFECTED ID IS : {}\n".format(key))


def DummyFuncUnknown(key):
    print("[+] ==> UNKNOWN USER ADDED TO BD. \n[+] AFFECTING ID : {} TO THE USER\n".format(key))


def updateEnroll(var: bool):
    var = True
    return var


def Calculate_Distance(top, right, bottom, left, method='Algebra'):
    if method == "Estimation":
        distancei = (2 * 3.14 * 180) / ((right - left) + (bottom - top) * 360) * 1000
        distance_in_cm = math.floor(distancei * 2.54)
    elif method == "Algebra":
        reye = left + ((right - left) / 2) - ((right - left) / 5)
        leye = left + ((right - left) / 2) + ((right - left) / 5)
        space = leye - reye
        f = 690
        r = 10
        distance = f * r / space
        distance_in_cm = int(distance)
    else:
        raise ValueError("method Value should be Estimation or Algebric")
    return distance_in_cm


def main_loop():
    face_counter = 0
    if running_on_jetson_nano():
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if model == 'cnn':
            face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        else:
            face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=number_jitters)

        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            metadata = find_faces(face_encoding)
            Just_Enrolled = False  # type: bool
            if metadata is not None:
                top, right, bottom, left = face_location
                dis = Calculate_Distance(top * 4, right * 4, bottom * 4, left * 4, method=Distance_Method)
                if dis < Distance_Threshhold:
                    DummyFuncEnrolled(metadata['id'])
                    face_label = 'ID :{}'.format(metadata['id'])
                else:
                    face_label = 'far'
                    print("\n")
            else:
                face_label = "Unknown"
                top, right, bottom, left = face_location
                dis = Calculate_Distance(top * 4, right * 4, bottom * 4, left * 4, method=Distance_Method)
                # print("dis = {}".format(dis))
                if dis < Distance_Threshhold:
                    face_image = small_frame[top:bottom, left:right]
                    face_image = cv2.resize(face_image, (150, 150))
                    if Auto_Enrollment == "Enabled":
                        just = register(face_encoding, face_image)
                        DummyFuncUnknown(just)
                        #updateEnroll()
                    else:
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            just = register(face_encoding, face_image)
                            DummyFuncUnknown(just)
                            #updateEnroll()
                            # break
                else:
                    print("\n")
            #if Just_Enrolled:
                #DummyFuncUnknown(metadata['id'])
            face_labels.append(face_label)

        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            distance_from_cam = Calculate_Distance(top, right, bottom, left, method=Distance_Method)
            if distance_from_cam < Distance_Threshhold:
                if 'Unknown' in face_label:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, 'Distance : ' + str(distance_from_cam) + ' cm',
                                (left + 140, top + (bottom - top) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (239, 246, 255))
                    cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (255, 255, 255), 1)

                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, 'Distance : ' + str(distance_from_cam) + ' cm',
                                (left + 140, top + (bottom - top) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (239, 246, 255))
                    cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (255, 255, 255), 1)
            else:
                cv2.putText(frame, "GET YOUR FACE NEAR THE CAM", (200, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255),
                            1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            backup()
            break

        if len(face_locations) > 0 and face_counter > 100:
            backup()
            face_counter = 0
        else:
            face_counter += 1

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    main_loop()
