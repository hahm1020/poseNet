import sys

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# 현재 파일(src)의 부모 디렉토리(프로젝트 루트)를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import PoseNet


# models/model.py에 정의한 PoseNet 모델을 임포트합니다.
#from models.model import PoseNet


def load_model(model_path, device):
    """
    저장된 모델 파라미터를 불러와 PoseNet 모델을 생성합니다.
    """
    model = PoseNet(num_keypoints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 평가 모드 전환
    return model


def preprocess_image(image_path):
    """
    테스트 이미지를 불러오고, 학습 시와 동일한 전처리를 적용합니다.
    """
    # 학습 시 사용한 전처리와 동일하게 적용
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform_pipeline(image).unsqueeze(0)  # 배치 차원 추가
    return input_tensor, image


def infer_keypoints(model, input_tensor, device):
    """
    모델에 이미지를 입력하여 예측된 keypoint를 반환합니다.
    출력은 (17, 3) 형태로 재구성됩니다.
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    # 모델 출력 shape: (1, 51) → reshape하여 (17, 3)
    keypoints = output.cpu().numpy().reshape(-1, 3)
    return keypoints


def visualize_results(image, keypoints):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    # 원본 이미지 크기 얻기
    width, height = image.size
    scale_x = width / 224
    scale_y = height / 224
    # 스케일링된 좌표로 시각화
    for (x, y, v) in keypoints:
        plt.scatter(x * scale_x, y * scale_y, c="red", marker="x")
    plt.title("Predicted Pose Keypoints")
    plt.axis("off")
    plt.show()



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 모델 로드
    model = load_model(args.model_path, device)
    print("모델 로드 완료.")

    # 2. 테스트 이미지 전처리
    input_tensor, original_image = preprocess_image(args.image_path)
    print("이미지 전처리 완료.")

    # 3. 모델 추론
    keypoints = infer_keypoints(model, input_tensor, device)
    print("추론 완료. 예측된 keypoints:")
    print(keypoints)

    # 4. 결과 시각화
    visualize_results(original_image, keypoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test PoseNet model on an image")
    parser.add_argument(
        '--model_path',
        type=str,
        default="models/posenet_model.pth",
        help="저장된 모델 checkpoint 경로"
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default="data/test_image.jpg",
        help="테스트할 이미지 경로"
    )
    args = parser.parse_args()
    main(args)
