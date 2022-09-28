from torchvision import datasets

dataset = datasets.CelebA("data", split="train", download=True,)
