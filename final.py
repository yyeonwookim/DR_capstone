import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F

# 하이퍼파라미터 설정
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_classes = 5  # DR 등급 (0-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 파일 경로 설정
train_dir = '/data/yeonwx/train_images/train/train6000'
val_dir = '/data/yeonwx/train_images/train/val2000'
test_dir = '/data/yeonwx/train_images/train/test2000'

# trainLabels.csv 파일 로드
labels_df = pd.read_csv('trainLabels.csv')
labels_df['image'] = labels_df['image'] + '.jpeg'

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
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 정의
train_dataset = RetinopathyDataset(train_dir, train_df, transform=train_transform)
val_dataset = RetinopathyDataset(val_dir, val_df, transform=test_transform)
test_dataset = RetinopathyDataset(test_dir, test_df, transform=test_transform)

# 클래스 가중치 조정
class_weights = [1.0, 2.0, 2.0, 1.5, 3.0]
sample_weights = [class_weights[label] for label in train_df['level']]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# EfficientNet 모델 로드 및 수정
model = EfficientNet.from_pretrained('efficientnet-b5')
model._fc = nn.Linear(model._fc.in_features, num_classes)
nn.init.xavier_uniform_(model._fc.weight)
nn.init.zeros_(model._fc.bias)
model = model.to(device)

# 손실 함수 정의
class DistanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0):
        super(DistanceWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.penalty_matrix = self._create_penalty_matrix()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def _create_penalty_matrix(self):
        penalty_matrix = torch.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                penalty_matrix[i, j] = abs(i - j)
        return penalty_matrix.to(device)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        penalties = self.penalty_matrix[targets]
        log_probs = -torch.log(probs + 1e-12)
        weighted_log_probs = penalties * log_probs
        distance_loss = torch.sum(targets_one_hot * weighted_log_probs, dim=1).mean()
        ce_loss = self.ce_loss(logits, targets)
        return self.alpha * ce_loss + distance_loss

criterion = DistanceWeightedCrossEntropyLoss(num_classes=num_classes)

# Optimizer와 Scheduler 정의
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1)

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

    accuracy = correct / total if total > 0 else 0.0
    mse = mean_squared_error(all_labels, all_preds)

    print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    return running_loss / len(dataloader.dataset), accuracy, all_labels, all_preds, mse

# 훈련 루프
best_val_loss = float('inf')
best_model_wts = model.state_dict()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy, _, _, val_mse = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val MSE: {val_mse:.4f}, LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()

# 최적 모델 로드
model.load_state_dict(best_model_wts)

# 테스트 평가
test_loss, test_accuracy, _, _, test_mse = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test MSE: {test_mse:.4f}")

