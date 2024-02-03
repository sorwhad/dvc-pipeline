from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

from torch.optim import Adam

from utils import Accumulator
import tensorboardX
import yaml
import os

import mlflow
from mlflow.models import infer_signature


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.dense1 = nn.Linear(800, 240)
        self.dense2 = nn.Linear(240, 168)
        self.dense3 = nn.Linear(168, 10)

    def forward(self, x, verbose=True):
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        pool1_out = self.pool1(relu1_out)
        if verbose:
            print(pool1_out.shape)

        conv2_out = self.conv2(pool1_out)
        relu2_out = self.relu2(conv2_out)
        pool2_out = self.pool2(relu2_out)
        if verbose:
            print(pool2_out.shape)
        flatten = pool2_out.view((-1, 800))

        dense1_out = self.dense1(flatten)
        relu3_out = self.relu3(dense1_out)

        dense2_out = self.dense2(relu3_out)
        relu4_out = self.relu4(dense2_out)

        logits = self.dense3(relu4_out)

        return logits
        


train_dataset = CIFAR10('./datasets/cifar10', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = CIFAR10('./datasets/cifar10', train=False, transform=transforms.ToTensor(), download=False)

with open('params.yaml', 'r') as fp:
    params = yaml.safe_load(fp)['train']
lenet = LeNet()

loss = nn.NLLLoss()


optimizer = Adam(lenet.parameters(), lr=params['lr'])


train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

log_softmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)


device = 'cpu'


experiment_name = f'{params["batch_size"]}-{params["num_epochs"]}-{params["lr"]}'
writer = tensorboardX.SummaryWriter(os.path.join('logs', experiment_name))


mlflow.set_tracking_uri('http://localhost:5005')

with mlflow.start_run() as run:
    mlflow.log_params(params)
    for epoch_num in range(params['num_epochs']):
        accumulator = Accumulator()

        lenet.train()
        for index, batch in enumerate(train_dataloader):
            X, y = batch

            batch_size = X.shape[0]
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = lenet(X, verbose=False)

            
            loss_value = loss(log_softmax(logits), y)
            loss_value.backward()

            accumulator.append(loss_value.item(), batch_size)

            optimizer.step()

        print(epoch_num, 'train', accumulator.get())


        writer.add_scalar('loss/train', accumulator.get(), global_step=epoch_num)

        mlflow.log_metric('loss/train', accumulator.get(), step=epoch_num)
        lenet.eval()
        val_accumulator = Accumulator()
        accuracy_accumulator = Accumulator()
        for index, batch in enumerate(val_dataloader):
            X, y = batch
            batch_size = X.shape[0]
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                logits = lenet(X, verbose=False)
                loss_value = loss(log_softmax(logits), y)
                val_accumulator.append(loss_value.item(), batch_size)
                probabilities = softmax(logits)
                class_labels = torch.argmax(probabilities, dim=1)
                accuracy = accuracy_score(
                    y_true=y.cpu().numpy(),
                    y_pred=class_labels.cpu().numpy()
                )
                accuracy_accumulator.append(accuracy, batch_size)


        print(epoch_num, 'val', val_accumulator.get())
        print(epoch_num, 'accuracy', accuracy_accumulator.get())

        writer.add_scalar('loss/val', val_accumulator.get(), global_step=epoch_num)
        writer.add_scalar('metrics/accuracy', accuracy_accumulator.get(), global_step=epoch_num)
        mlflow.log_metric('loss/val', val_accumulator.get(), step=epoch_num)
        mlflow.log_metric('metrics/accuracy', accuracy_accumulator.get(), step=epoch_num)
        writer.flush()

        checkpoint_folder = os.path.join('checkpoints', experiment_name)

        os.makedirs(checkpoint_folder, exist_ok=True)

        torch.save(lenet.state_dict(), os.path.join(checkpoint_folder, f'{epoch_num}.pt'))
        mlflow.pytorch.log_model(   
            lenet,
            artifact_path=f'model-{epoch_num}',
            signature=infer_signature(X.numpy(), logits.numpy()),
            input_example=X.numpy(),
        )

