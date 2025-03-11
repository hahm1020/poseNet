import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse

from models.model import PoseNet
from dataset import PoseDataset, load_coco_keypoints

def train(args):
    # 디바이스 설정 (GPU 사용 가능하면 GPU로)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.learning_rate

    # COCO train 데이터셋 로드
    train_image_paths, train_keypoints = load_coco_keypoints(data_dir, data_type='train2017')
    print(f"Train 데이터 개수: {len(train_image_paths)}")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = PoseDataset(train_image_paths, train_keypoints, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train 데이터로더 배치 수: {len(train_loader)}")

    # COCO validation 데이터셋 로드
    val_image_paths, val_keypoints = load_coco_keypoints(data_dir, data_type='val2017')
    print(f"Validation 데이터 개수: {len(val_image_paths)}")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_dataset = PoseDataset(val_image_paths, val_keypoints, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation 데이터로더 배치 수: {len(val_loader)}")

    # 모델, 손실함수, 옵티마이저 정의
    model = PoseNet(num_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("모델, 손실함수, 옵티마이저 초기화 완료")

    # 학습 루프
    print("학습 시작...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, keypoints) in enumerate(train_loader):
            try:
                images = images.to(device)
                keypoints = keypoints.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                
                # 10개의 배치마다 현재 진행상황 출력 (더 자주 출력하도록 수정)
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"학습 중 에러 발생: {str(e)}")
                print(f"배치 인덱스: {i}, 이미지 shape: {images.shape}, 키포인트 shape: {keypoints.shape}")
                raise e
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 완료")
        print(f"평균 Training Loss: {epoch_loss:.4f}")

        # 검증 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, keypoints in val_loader:
                images = images.to(device)
                keypoints = keypoints.to(device)
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    # 모델 저장
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "posenet_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PoseNet on COCO Keypoints")
    parser.add_argument('--data_dir', type=str, default='data', help='COCO 데이터셋 루트 디렉토리 (train2017, val2017, annotations 포함)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 사이즈')
    parser.add_argument('--num_epochs', type=int, default=10, help='학습 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='학습률')
    parser.add_argument('--save_dir', type=str, default='models', help='학습된 모델을 저장할 디렉토리')
    args = parser.parse_args()

    train(args)
