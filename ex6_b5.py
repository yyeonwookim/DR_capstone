import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from PIL import Image, UnidentifiedImageError
from utils.losses import CostSensitiveLoss  # Cost-Sensitive Loss 임포트

# 하이퍼파라미터 설정
batch_size = 16
num_epochs = 10  # 디버깅을 위한 epoch 수 감소
learning_rate = 0.001
num_classes = 5  # DR 등급 (0-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 파일 경로 설정
train_dir = '/data/yeonwx/train_images/train/train6000'
val_dir = '/data/yeonwx/train_images/train/val2000'
test_dir = '/data/yeonwx/train_images/train/test2000'

# trainLabels.csv 파일 로드
labels_df = pd.read_csv('trainLabels.csv')
labels_df['image'] = labels_df['image'] + '.jpeg'  # 확장자 추가

# 이미지 파일 이름 확인
train_images = os.listdir(train_dir)
val_images = os.listdir(val_dir)
test_images = os.listdir(test_dir)

train_df = labels_df[labels_df['image'].isin(train_images)]
val_df = labels_df[labels_df['image'].isin(val_images)]
test_df = labels_df[labels_df['image'].isin(test_images)]

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# 데이터셋 클래스 정의
class RetinopathyDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No such file: '{img_path}'")

        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            raise FileNotFoundError(f"Unable to open image file: {img_path}")

        if self.transform:
            image = self.transform(image)

        label = self.labels_df.iloc[idx, 1]
        return image, label

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 정의
train_dataset = RetinopathyDataset(train_dir, train_df, transform=transform)
val_dataset = RetinopathyDataset(val_dir, val_df, transform=transform)
test_dataset = RetinopathyDataset(test_dir, test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# EfficientNet 모델 로드 및 수정
model = EfficientNet.from_pretrained('efficientnet-b5')
model._fc = nn.Linear(model._fc.in_features, num_classes)
model = model.to(device)

# Cost-Sensitive Loss 정의
criterion = CostSensitiveLoss(n_classes=num_classes)

# 비용 행렬 설정 (먼 등급에 더 높은 비용 부여)
criterion.M = torch.tensor([
    [0.0, 0.5, 1.0, 1.5, 2.0],
    [0.5, 0.0, 0.5, 1.0, 1.5],
    [1.0, 0.5, 0.0, 0.5, 1.0],
    [1.5, 1.0, 0.5, 0.0, 0.5],
    [2.0, 1.5, 1.0, 0.5, 0.0]
]).to(device)

# 옵티마이저 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 훈련 및 평가 함수 정의
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = accuracy_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)

    return running_loss / len(dataloader.dataset), accuracy, mse, correct, total

# 훈련 루프
best_val_accuracy = 0.0
best_model_wts = model.state_dict()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy, val_mse, val_correct, val_total = evaluate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f} ({val_correct}/{val_total}), Val MSE: {val_mse:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = model.state_dict()

# 최적 모델 로드
model.load_state_dict(best_model_wts)

# 테스트 평가
test_loss, test_accuracy, test_mse, test_correct, test_total = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_total}), Test MSE: {test_mse:.4f}")

# 최종 모델 저장
torch.save(model.state_dict(), 'best_efficientnet_retinopathy.pth')

#cat logs/slurm-65537.out
'''
Training samples: 6372
Validation samples: 2000
Test samples: 1992
Loaded pretrained weights for efficientnet-b5
Epoch [1/10], Train Loss: 0.2752, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [2/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [3/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [4/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [5/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [6/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [7/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [8/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [9/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Epoch [10/10], Train Loss: 0.2629, Val Loss: 0.2705, Val Accuracy: 0.7320 (1464/2000), Val MSE: 1.2820
Test Loss: 0.2324, Test Accuracy: 0.7565 (1507/1992), Test MSE: 1.0271
'''