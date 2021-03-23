import time
import cv2
import numpy as np
import redis
import pickle

r = redis.Redis(host='test_p', port=6379, db=0)

images = []
path1 = "0002_c0_f0000001.jpg"
path2 = "0002_c0_f0000003.jpg"
path3 = "0002_c0_f0000005.jpg"

im1 = np.moveaxis(cv2.resize(cv2.imread(path1), (256,128)), -1, 0) / 255.
im2 = np.moveaxis(cv2.resize(cv2.imread(path2), (256,128)), -1, 0) / 255.
im3 = np.moveaxis(cv2.resize(cv2.imread(path3), (256,128)), -1, 0) / 255.
print(im3.shape)

job1 = np.array([im1, im2])
job2 = np.array([im2, im3])
job3 = np.array([im1, im3])

for key in r.scan_iter("*"):
    r.delete(key)

while True:
    print("ping pong: START!")
    key_arr = np.array(["jb1", "jb2", "jb3"])
    print("ping pong: Try to dump all jobs (jb1, jb2) to redis!")

    r.set("jb1", pickle.dumps(job1, 2))
    r.set("jb2", pickle.dumps(job2, 2))
    r.set("jb3", pickle.dumps(4, 2)) # error

    print("ping pong: DONE!")
    print("ping pong: send request to reid module!")
    r.lpush('request5', pickle.dumps(key_arr, 2))
    print("ping pong: waiting response!")
    response_list_keys = r.brpop('response5')
    print("ping pong: DONE!")
    resp_key_list, error_key_list = pickle.loads(response_list_keys[1])
    for key in resp_key_list:
        print(key)
        p_ans = r.get(key)
        ans = pickle.loads(p_ans)
        print(ans.shape, type(ans))

    for key in error_key_list:
        p_ans = r.get(key)
        ans = pickle.loads(p_ans)
        print(key, ans)

    print("ping pong: FINISH!")
    time.sleep(5)
