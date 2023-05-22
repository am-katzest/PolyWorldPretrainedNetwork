import torch
from skimage import io
from skimage.transform import resize


def loadSample(name):
    window_size = 320
    image = io.imread(name)
    image = resize(
        image,
        (window_size, window_size, 3),
        anti_aliasing=True,
        preserve_range=True,
    )
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1) / 255.0
    return torch.unsqueeze(image, 0).float()
