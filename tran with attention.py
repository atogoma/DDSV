import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os

data_dir = 'exp1/png250116'
batch_size = 32
input_size = 224 
num_epochs = 15
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_transform = transforms.Compose([
	transforms.Grayscale(num_output_channels=3),
	transforms.Resize((input_size, input_size)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(15),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
	transforms.Grayscale(num_output_channels=3),
	transforms.Resize((input_size, input_size)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
test_dataset.dataset.transform = val_transform  


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = BinaryClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def train_model():
	best_acc = 0.0
	for epoch in range(num_epochs):
		model.train()
		running_loss = 0.0
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.float().unsqueeze(1).to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * inputs.size(0)

		epoch_loss = running_loss / len(train_dataset)

		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in test_loader:
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)
				predicted = (torch.sigmoid(outputs) > 0.5).float()
				total += labels.size(0)
				correct += (predicted.squeeze() == labels).sum().item()

		acc = correct / total
		print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f}')

		if acc > best_acc:
			best_acc = acc
			torch.save(model.state_dict(), 'best_model.pth')

	print(f'Best Test Acc: {best_acc:.4f}')


if __name__ == '__main__':
	train_model()