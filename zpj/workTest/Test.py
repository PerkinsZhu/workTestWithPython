import requests
import json, time
import os
from threading import Thread


def doWork(fileName):
    url = 'http://api.learningpal.com/math/upload'
    files = {'file': open('./imgs/' + fileName, 'rb')}
    # files = {'file': open('./' + fileName, 'rb')}
    headers = {'content-type': 'application/json'}
    try:
        r = requests.post(url, files=files)
        response = json.loads(r.text)
        task_ID = response['task_ID']
        payload = {'task_ID': task_ID, 'password': ""}
        url2 = 'http://api.learningpal.com/math/result'
        while True:
            response = requests.post(url2, data=json.dumps(payload), headers=headers)
            if response.text and "processing" not in response.text:
                break
    except Exception as err:
        print("==="+err)
    print(fileName+"    ---->   "+json.loads(response.text).get("result"))


def start():
    for root, dirs, files in os.walk("imgs"):
        for file in files:
            t = Thread(target=doWork(file))
            t.start()


if __name__ == '__main__':
    # start()
    doWork("test7.png")