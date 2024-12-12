import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.model import CustomefficientnetV2M
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os



trasform = transforms.Compose([
                                transforms.Resize((5120, 5120)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


data_set = torchvision.datasets.ImageFolder('data/training_set/', transform=trasform)
train_dataset, test_dataset = torch.utils.data.random_split(data_set, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=4)


print(len(train_loader))
images, labels = next(iter(train_loader))
print(data_set.classes)
#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# loss metric and optimizer
model, optimizer = CustomefficientnetV2M(num_classes=2, pretrained=True, fixed_feature_extr=True)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


EPOCHS = 15

best_vloss = 1_000_000.
epoch_number = 0
for epoch in range(EPOCHS):

    model.train(True)

    running_loss = 0.
    avg_loss = 0.
    pbar = tqdm(enumerate(train_loader), total = len(train_loader),  desc=f"{epoch} epoch", unit="item")
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            avg_loss = running_loss / 1000 # loss per batch
            tb_x = epoch_number * len(train_loader) + i + 1
            writer.add_scalar('Loss/train', avg_loss, tb_x)
            running_loss = 0.
            pbar.set_postfix_str(f"Training loss {avg_loss:,.2f}")

    running_vloss = 0.0
    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"{epoch} epoch", unit="item")
    with torch.no_grad():
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_vloss += loss
            pbar.set_postfix_str(f"Testing loss {running_vloss/len(test_loader):,.2f}")

    avg_vloss = running_vloss / (i + 1)

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), f'checkpoints/model_best_epoch{epoch}.pt')

    torch.save(model.state_dict(), f'checkpoints/model_last_epoch{epoch}.pt')
    epoch_number += 1




classes = data_set.classes
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')