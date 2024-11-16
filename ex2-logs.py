import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, UnidentifiedImageError

# 하이퍼파라미터 설정
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_classes = 5  # DR 등급 (0-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 여러 파트로 나뉜 zip 파일을 하나로 결합하는 함수
def combine_zip_parts(zip_parts, output_zip):
    with open(output_zip, 'wb') as output_file:
        for part in zip_parts:
            with open(part, 'rb') as part_file:
                output_file.write(part_file.read())
    print(f"Combined zip parts into {output_zip}")

# 파일 경로 설정
zip_dir = './'  # 파트 파일이 저장된 디렉토리
train_parts = [f"{zip_dir}train.zip.{i:03}" for i in range(1, 6)]
test_parts = [f"{zip_dir}test.zip.{i:03}" for i in range(1, 8)]
combined_train_zip = './train.zip'
combined_test_zip = './test.zip'

# train.zip 파일 결합
combine_zip_parts(train_parts, combined_train_zip)

# test.zip 파일 결합
combine_zip_parts(test_parts, combined_test_zip)

# 압축 해제 디렉토리 설정
train_extract_dir = './train_images/train'
test_extract_dir = './test_images'

# 결합된 파일 압축 해제
with zipfile.ZipFile(combined_train_zip, 'r') as zip_ref:
    zip_ref.extractall(train_extract_dir)
print("Extracted training images.")

with zipfile.ZipFile(combined_test_zip, 'r') as zip_ref:
    zip_ref.extractall(test_extract_dir)
print("Extracted test images.")

# trainLabels.csv 파일 로드
labels_df = pd.read_csv('./trainLabels.csv.zip', compression='zip')
print("Loaded labels from CSV.")

# 학습 및 검증 세트 분할
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
print(f"Split data into {len(train_df)} training samples and {len(val_df)} validation samples.")

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
        
        img_path_jpeg = os.path.join(self.image_dir, f"{img_name}.jpeg")
        
        # 파일 존재 여부 확인
        if os.path.exists(img_path_jpeg):
            img_path = img_path_jpeg
        else:
            raise FileNotFoundError(f"No such file: '{img_path_jpeg}'")

        # 이미지 로드 (예외 처리 추가)
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
print("Data transformation settings completed.")

# 데이터셋 및 데이터로더 정의
train_dataset = RetinopathyDataset(train_extract_dir, train_df, transform=transform)
val_dataset = RetinopathyDataset(train_extract_dir, val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Datasets and dataloaders defined.")

# EfficientNet 모델 로드 및 수정
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)  # 마지막 레이어 수정
model = model.to(device)
print("EfficientNet model loaded and modified.")

# 손실 함수와 옵티마이저 설정
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.2, 2.5], device=device)  # 클래스 가중치 설정
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Loss function and optimizer set up.")

# 커스텀 손실 함수 정의
def custom_loss(outputs, labels):
    # CrossEntropyLoss (분류 손실)
    ce_loss = criterion(outputs, labels)
    
    # MSE Loss (예측 등급과 실제 등급 차이에 대한 패널티)
    preds = torch.argmax(outputs, dim=1).float()
    labels = labels.float()  # 레이블도 float 타입으로 변환
    mse_criterion = nn.MSELoss()
    mse_loss = mse_criterion(preds, labels)
    
    # 손실 결합
    total_loss = ce_loss + mse_loss * 0.5  # MSE 손실에 0.5 가중치를 부여
    return total_loss

# 훈련 및 평가 루프 정의
def train_epoch(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = custom_loss(outputs, labels)  # 커스텀 손실 함수 사용
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)  # 기본 크로스 엔트로피 손실 사용
            running_loss += loss.item() * images.size(0)
            
            # Accuracy calculation
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(dataloader.dataset), accuracy

# 훈련 루프
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, val_accuracy = evaluate(model, val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

print("Training loop completed.")

# 모델 저장
torch.save(model.state_dict(), 'efficientnet_retinopathy.pth')
print("Model saved as 'efficientnet_retinopathy.pth'.")
