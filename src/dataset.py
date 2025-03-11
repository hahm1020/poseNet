import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from pycocotools.coco import COCO

class PoseDataset(Dataset):
    def __init__(self, image_paths, keypoints, transform=None):
        """
        image_paths: 이미지 파일 경로 리스트
        keypoints: 각 이미지에 해당하는 keypoint 좌표 (torch.Tensor 형태)
        transform: 이미지 전처리 (예: torchvision.transforms)
        """
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        kp = self.keypoints[idx]
        return image, kp

def load_coco_keypoints(data_dir, data_type='train2017'):
    """
    COCO 데이터셋의 어노테이션 파일을 읽어서,
    이미지 경로와 keypoint 정보를 추출합니다.

    data_dir: 데이터셋의 루트 디렉토리 (예: 'data')
    data_type: 'train2017' 또는 'val2017'
    """
    ann_file = os.path.join(data_dir, 'annotations', f'person_keypoints_{data_type}.json') # /disk1/coco/labels/coco-pose/annotations
    coco = COCO(ann_file)

    # 'person' 카테고리의 ID를 가져옵니다.
    cat_ids = coco.getCatIds(catNms=['person'])
    # 해당 카테고리에 속하는 이미지 ID를 가져옵니다.
    img_ids = coco.getImgIds(catIds=cat_ids)
    images = coco.loadImgs(img_ids)

    image_paths = []
    keypoints = []
    for img in images:
        # file_path = os.path.join(data_dir, data_type, img['file_name'])
        file_path = os.path.join('/disk1/coco/labels/coco-pose/images', data_type, img['file_name'])
        # 이미지에 해당하는 어노테이션 ID를 로드 (여러 인물이 있을 수 있음)
        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        # 여기서는 첫 번째 인물의 keypoints를 사용합니다.
        ann = anns[0]
        # keypoints는 [x1,y1,v1, x2,y2,v2, ..., x17,y17,v17] 형식입니다.
        kps = ann['keypoints']

        # 만약 시각화나 평가에 visibility 정보가 필요없다면, x와 y만 사용할 수 있음
        # 예: kps = [k for i, k in enumerate(kps) if i % 3 != 2]
        kps_tensor = torch.tensor(kps, dtype=torch.float32)

        image_paths.append(file_path)
        keypoints.append(kps_tensor)

    return image_paths, keypoints
