import pickle
import numpy as np
import torch
import torch.nn as nn
import redis
import torchreid
from torchreid.models.plr_osnet import PLR_OSNet

def main():
    model = PLR_OSNet(12345, fc_dims=512)

    torchreid.utils.load_pretrained_weights(model, "model.pth.tar-120")
    model = nn.DataParallel(model).cuda().eval()

    r = redis.Redis(host='test_p', port=6379, db=0)
    while True:
        request_list_keys = r.brpop('request5')
        get_np = pickle.loads(request_list_keys[1])

        resp_key_list = []
        error_key_list = []
        for key in get_np:
            try:
                bimages = r.get(key)
                images = pickle.loads(bimages)
                images = np.array(images)
                cuda_images = torch.from_numpy(images).cuda().float()

                with torch.no_grad():
                    ans = model(cuda_images).cpu().numpy()
                r.set("r" + key[1:], pickle.dumps(ans, 2), ex=None, px=None, nx=False, xx=False)
                resp_key_list.append("r" + key[1:])
            except Exception as e:
                r.set("e" + key[1:], pickle.dumps(e, 2), ex=None, px=None, nx=False, xx=False)
                error_key_list.append("e" + key[1:])
                continue
        resp_key_numpy = np.array([resp_key_list, error_key_list])
        r.lpush('response5',  pickle.dumps(resp_key_numpy, 2))



if __name__ == '__main__':
    main()
