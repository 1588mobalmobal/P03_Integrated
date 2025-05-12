import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import albumentations as A
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class_mapping = {
    0 :'void',
    1 :'dirt',
    # 2 :'sand',
    2 :'grass' ,
    3 :'tree' ,
    4 :'obstacle' ,
    5 :'water' ,
    6 :'sky' ,
    7 :'vehicle' ,
    8 :'person' ,
    9 :'hard_surface',
    10 : 'gravel',
    11 :'vegetation' ,
    # 13 :'mulch',
    12 :'rock',
    13 :'cannon',
}

# 분류된 클래스별로 색상을 할당합니다 
class_to_rgb_map = {
    0 : (0, 0, 0),
    1 : (108, 64, 20),
    # 2 : (255, 229, 204),
    2 : (0, 102, 0),
    3 : (0, 255, 0),
    4 : (0, 153, 153),
    5 : (0, 128, 255),
    6 : (0, 0, 255),
    7 : (255, 255, 0),
    8 : (255, 0, 127),
    9 : (64, 64, 64),
    10 : (100, 110, 50),
    11 : (183, 255, 0),
    # 13 : (153, 76, 0),
    12 : (160, 160, 160),
    13 : (140,120,240)
}

# 소스 이미지를 읽어서 클래스를 분류합니다 
rgb_to_class_map = {
    (0, 0, 0): 0, 
    (108, 64, 20): 1, 
    # (255, 229, 204): 2, 
    (255, 229, 204): 1, 
    (0, 102, 0): 2, 
    (0, 255, 0): 3, 
    (0, 153, 153): 4, 
    (0, 128, 255): 5, 
    (0, 0, 255): 6, 
    (255, 255, 0): 7, 
    (255, 0, 127): 8, 
    (64, 64, 64): 9, 
    (255, 128, 0): 10, 
    (255, 0, 0): 4, 
    # (153, 76, 0): 13, 
    (153, 76, 0): 1, 
    (102, 102, 0): 10, 
    (102, 0, 0): 11, 
    (0, 255, 128): 7, 
    (204, 153, 255): 8, 
    (102, 0, 204): 4, 
    (255, 153, 204): 11, 
    (0, 102, 102): 4, 
    (153, 204, 255): 12, 
    (102, 255, 255): 4, 
    (101, 101, 11): 4, 
    (114, 85, 47): 4,
    (170, 170, 170): 4,
    (41, 121, 255): 4,
    (101, 31, 255): 9,
    (137, 149, 9): 9,
    (134, 255, 239): 1,
    (99, 66, 34): 1,
    (110, 22, 138): 4,
    (140,120,240): 13,
    (183, 255, 0): 11,
    (100, 110, 50): 10,
    (183, 255, 0): 11,
    (160, 160, 160): 12
    }

directory = os.getcwd()
device = "cuda" if torch.cuda.is_available() else "cpu"

left_dir = '/mnt/c/Users/kbh11/OneDrive/Documents/Tank Challenge/capture_images/L'
right_dir = '/mnt/c/Users/kbh11/OneDrive/Documents/Tank Challenge/capture_images/R'
result_dir = "results"

def init_model():
    # mean= [-0.02662486, -0.01916305, -0.00590634]
    # std= [0.07481168, 0.07667251, 0.07697445]
    preprocessor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
                                                            size={"height": 512, "width": 512}, do_reduce_labels=False,
                                                            
                                                            )
    model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            num_labels=len(class_mapping),
            ignore_mismatched_sizes=True
        )
    # 학습된 가중치 로드
    model.config.image_size = 512
    model.decode_head.classifier = torch.nn.Conv2d(256, 14, kernel_size=1)
    state_dict = torch.load(os.path.join(directory, "segformer_b1_sim_only_augmented_alpha_combine_epoch_99.pth"))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
    # model.eval()  # 추론 모드 활성화
    print('😊 Segformer_b1 has been Initialized!!💙')
    # print(preprocessor.__dict__)
    return model, preprocessor

def predict_segmentation(image_path, model, preprocessor):
    sample_image = np.array(Image.open(image_path))
    inputs = preprocessor(images=sample_image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 25, H, W]
        # probs = torch.softmax(logits, dim=1)
        # print(f"Probs shape: {probs.shape}")  # [batch_size, 25, 512, 512]
        # for class_idx in range(14):
        #     class_prob = probs[0, class_idx, 0, 0]  # 첫 번째 픽셀의 클래스별 확률
        #     print(f"Class {class_idx} probability: {class_prob.item()}")
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # [1, 512, 512]
        prediction = predictions[0]
    
    return prediction


# 4. 시각화 함수 (클래스 인덱스 → RGB)
def visualize_segmentation(image_path, prediction, output_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # 모델 입력 크기에 맞춤
    
    # 클래스 인덱스를 RGB로 변환
    seg_map = np.zeros((512, 512, 3), dtype=np.uint8)
    color_array = np.array([class_to_rgb_map.get(i, (0, 0, 0)) for i in range(14)], dtype=np.uint8)
    seg_map = color_array[prediction]

    cv2.imwrite(output_path, seg_map)

    # unique_classes = np.unique(prediction)
    # legend_elements = [
    #         Patch(facecolor=np.array(class_to_rgb_map[idx])/255, label=class_mapping.get(idx, f"Class {idx}"))
    #         for idx in unique_classes if idx in class_to_rgb_map
    #     ]

    # # 원본 이미지와 세그멘테이션 맵 시각화

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ax1.imshow(image)
    # ax1.set_title("Original Image")
    # ax1.axis("off")

    # ax2.imshow(seg_map)
    # ax2.set_title("Segmentation Map")
    # ax2.axis("off")

    # # 범례 추가
    # ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # fig.tight_layout()
    # plt.savefig(output_path, format="png", bbox_inches="tight")
    # plt.close(fig)

def get_item_dir(from_dir):
    items = sorted(os.listdir(from_dir))
    item = items[-1]
    item_dir = os.path.join(from_dir, item)
    return items, item_dir


def get_vehicle_distance(seg_model, image_processor):
    left_items, left_item_dir = get_item_dir(left_dir)
    right_items, right_item_dir = get_item_dir(right_dir)

    left_prediction = predict_segmentation(left_item_dir, seg_model, image_processor)
    right_prediction = predict_segmentation(right_item_dir, seg_model, image_processor)

    img_left = cv2.imread(left_item_dir, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_item_dir, cv2.IMREAD_GRAYSCALE)

    seg_mask_left_resized = cv2.resize(left_prediction.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    seg_mask_right_resized = cv2.resize(right_prediction.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

    # 3. 스테레오 매칭 설정 (간단한 SGBM 설정)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 1 * 5**2,
        P2=32 * 1 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )

    # 4. 시차 맵 계산
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

    # 시차 맵 정규화 (시각화용)
    # disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 5. 깊이 맵 계산
    focal_length = 1050  # 초점 거리 (픽셀 단위, 예시 값)
    baseline = 1        # 기본선 거리 (미터 단위, 예시 값)
    depth_map = np.zeros_like(disparity)
    valid_disparity = disparity > 0.1  # 유효한 시차만 선택
    depth_map[valid_disparity] = (focal_length * baseline) / disparity[valid_disparity]

    # 깊이 맵 정규화 (시각화용)
    # depth_map_visual = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 6. 특정 클래스(예: 클래스 1)에 대한 깊이 맵 추출
    target_class = 7
    mask_left = (seg_mask_left_resized == target_class).astype(np.uint8)
    mask_right = (seg_mask_right_resized == target_class).astype(np.uint8)
    combined_mask = cv2.bitwise_and(mask_left, mask_right)  # 좌/우 공통 영역
    masked_depth_map = cv2.bitwise_and(depth_map, depth_map, mask=combined_mask)

    distance = np.mean(masked_depth_map[masked_depth_map != 0])

    for item in left_items:
        os.remove(os.path.join(left_dir, item))
    for item in right_items:
        os.remove(os.path.join(right_dir, item))

    return distance