# my-model

이 프로젝트는 PyTorch를 사용하여 이미지로부터 사람의 포즈 keypoint를 추출하는 PoseNet 모델을 구현하는 예제입니다.

## 폴더 구조
````
my-model/ 
├── data/ # 학습 및 검증 이미지 및 어노테이션 저장 폴더 
├── models/ 
│ ├── init.py 
│ └── model.py # PoseNet 모델 정의 
├── src/ 
│ ├── init.py 
│ ├── dataset.py # PoseDataset 클래스 정의 
│ └── train.py # 학습 및 검증 코드 
├── requirements.txt # 필요한 라이브러리 목록 
└── README.md # 프로젝트 개요 및 사용법
````

## 설치 및 실행

1. **COCO 데이터셋 다운로드 및 배치**
    - [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
    - [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
    - [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)  
      위 파일들을 다운로드한 후, 압축을 풀어 `data/` 폴더 아래에 아래와 같이 배치합니다.  


2. **패키지 설치**

 ```bash
 pip install -r req.txt
 ```

3. **학습 실행**

 ```bash
 python src/train.py --data_dir data --batch_size 32 --num_epochs 10 --learning_rate 1e-3 --save_dir models
 ```

학습이 완료되면 `models/posenet_model.pth`에 학습된 모델이 저장됩니다.


4. **추론 실행**
```bash
 python src/test_pose.py --model_path models/posenet_model.pth --image_path data/{test-img}
```
