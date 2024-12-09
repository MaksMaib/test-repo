
import torch
import torchvision
from models.model import CustomefficientnetV2M
from torchvision import transforms


trasform = transforms.Compose([
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


data_set = torchvision.datasets.ImageFolder('data/test_set/', transform=trasform)

eval_loader = torch.utils.data.DataLoader(data_set,
                                          batch_size=4,
                                          shuffle=False,
                                          num_workers=4)


print(len(eval_loader))
images, labels = next(iter(eval_loader))
print(data_set.classes)
#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, optimizer = CustomefficientnetV2M(num_classes=2, pretrained=False, fixed_feature_extr=False)
model.load_state_dict(torch.load('checkpoints/model_best_epoch14.pt', weights_only=True))
model.to(device)
model.eval()

classes = data_set.classes
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in eval_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
print('Best model statistic')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')