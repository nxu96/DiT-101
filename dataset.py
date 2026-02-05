import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, PILToTensor


# The old classic MNIST dataset.
class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        # MNIST dataset returns PIL images by default.
        self.ds = torchvision.datasets.MNIST("./mnist/", train=is_train, download=True)
        # Compose the transform pipeline.
        self.img_convert = Compose(
            [
                PILToTensor(),  # Convert PIL image to tensor.
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return self.img_convert(img) / 255.0, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = MNIST()
    # 60,000 samples with integer labels.
    # Shape of each image is (1, 28, 28).
    # Image type is torch.float32.
    print(f"Number of samples: {len(ds)}")
    img, label = ds[0]
    print("Image label is ", label)
    print("Image shape is ", img.shape)
    print("Image type is ", img.dtype)
    print("Image device is ", img.device)
    print(f"Image value range is {img.min()} to {img.max()}")  # 0.0 to 1.0.
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("mnist.png")
