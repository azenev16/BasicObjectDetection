import cv2
from playsound import playsound

#img = cv2.imread('WIN_20210723_17_45_58_Pro.JPG')




classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames =f.read().rstrip('\n').split('\n')


configPath ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1. / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, draw= True,objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5, nmsThreshold=0.2)
   # print(classIds)
    if len(objects) == 0: objects = classNames
    objectInfo= []
    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId-1], (box[0]+10, box[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(img,str(round(confidence*100,2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    playsound(r"C:\Users\Veneza Dayao\PycharmProjects\ObjectDetection\cellphone.mp3")



    return  img, objectInfo



if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 648)
    cap.set(4, 480)

    while True:
        check, img = cap.read()
        result, objectInfo = getObjects(img, objects=['cell phone'])
        cv2.imshow("Output", img)
        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()


