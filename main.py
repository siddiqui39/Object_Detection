import cv2

# loading the class
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# loading the model
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)


# detecting object
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox

        class_name = classes[class_id]
        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (200, 0, 50), 3)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
