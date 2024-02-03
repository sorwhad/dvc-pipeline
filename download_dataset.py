from torchvision.datasets import CIFAR10
from torchvision import transforms

train_dataset = CIFAR10('./datasets/cifar10', train=True, transform=transforms.ToTensor(), download=True)
