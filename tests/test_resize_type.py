import torch
import numpy as np
from tqdm import tqdm
import time
from scipy.ndimage.interpolation import zoom


if __name__ == '__main__':
    image = np.random.random(size=(100, 100, 100))
    target_size = (200, 200, 200)
    ratio = np.asarray([dst / src for dst, src in zip(target_size, image.shape)])
    print(ratio)
    time_count = []
    for i in tqdm(range(50)):
        start = time.time()
        resized_image = torch.nn.functional.interpolate(torch.from_numpy(image[None, None, ...]),
                                                        size=target_size,
                                                        mode='trilinear').squeeze().detach().numpy()
        end = time.time()
        time_count.append(end - start)

    print("[INFO] Torch spend time: {}".format(np.asarray(time_count).mean()))
    time_count.clear()
    for i in tqdm(range(50)):
        start = time.time()
        resized_image = zoom(image, ratio, order=1)
        end = time.time()
        time_count.append(end - start)
    print("[INFO] Scipy spend time: {}".format(np.asarray(time_count).mean()))

