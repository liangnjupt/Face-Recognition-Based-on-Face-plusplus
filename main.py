import requests
from json import JSONDecoder
import cv2
import os
import time
detect_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
key = "MTRTorAx4tkJx4W2BqUiLX8JQgYJ1L1f"
secret = "lQ3tYF5415OhCJDzx9JZQ1yl2_mLz0kv"

def face_detect(frame):
    img_encode = cv2.imencode('.jpg', frame)[1]
    data = {"api_key": key, "api_secret": secret, "return_landmark": "1", "return_attributes": "gender"}
    files = {"image_file": img_encode}
    response = requests.post(detect_url, data=data, files=files)
    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    face_rectangles = []
    for face in req_dict['faces']:
        if 'face_rectangle' in face.keys():
            face_rectangles.append(face['face_rectangle'])
    return face_rectangles

def face_recognition(frame,face_rectangles,templatepath="./dataset/template",draw_text=True):
    data = {"api_key": key, "api_secret": secret}
    draw_information=[]
    for i in face_rectangles:
        w=i['width']
        t=i['top']
        l=i['left']
        h=i['height']
        # cv2.imshow("gag",frame[t:t+h,l:l+w,:])
        # cv2.waitKey(0)
        face1=cv2.imencode('.jpg', frame[t:t+h,l:l+w,:])[1]
        cv2.rectangle(frame, (l, t), (w + l, h + t), (0, 0, 255), 2)  # opencv的标框函数
        for j in os.listdir(templatepath):
            tempath=os.path.join(templatepath,j)
            face2 = cv2.imencode('.jpg', cv2.imread(tempath))[1]
            files = {"image_file1": face1, "image_file2": face2}
            response = requests.post(compare_url, data=data, files=files)
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)

            if 'confidence' in req_dict.keys():
                confindence = req_dict['confidence']
                if confindence >= 65:
                    draw_information.append([j.split(".")[0],(l,t)])
    if draw_text and len(draw_information)>0:
        for x in draw_information:
            cv2.putText(frame,x[0],x[1],cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    return frame

if __name__=="__main__":
    filepath="./dataset/sample/test.jpg"
    frame = cv2.imread(filepath)
    face_rectangles=face_detect(frame)
    frame=face_recognition(frame,face_rectangles,templatepath="./dataset/template",draw_text=True)
    cv2.imshow("nizhencai",frame)
    cv2.waitKey(0)
