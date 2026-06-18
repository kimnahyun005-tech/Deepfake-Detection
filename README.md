# <h1 align="center"> RGB & Artifact-based Deepfake Detection with CNN Ensemble</h1>

<p align="center">
RGB 및 Artifact 앙상블 기반 딥페이크 탐지 모델 개발 프로젝트
</p>

## 1. Project Overview

본 프로젝트는 CNN 기반 딥러닝 모델을 활용하여 이미지의 딥페이크 여부를 탐지하는 것을 목표로 한다.

RGB 이미지와 Artifact Map을 각각 활용하여 탐지 성능을 비교하고, 두 정보를 결합한 앙상블 기법을 적용하여 탐지 성능을 비교·분석하고, 최적의 탐지 모델을 도출하고자 하였다.

## 2. Features
- RGB 이미지 기반 딥페이크 탐지
- Artifact Map 기반 딥페이크 탐지
- RGB + Artifact Ensemble 기반 딥페이크 탐지
- EfficientNet-B4, MobileNetV3, ResNet-50, ViT-B/16 성능 비교
- 입력 해상도에 따른 모델 성능 비교
- Confusion Matrix 기반 성능 분석
- Streamlit 기반 딥페이크 탐지 웹 서비스 구현
- Vision Transformer(16x16 patch) 기반 추가 실험 수행

## 3. Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB)
![PyYAML](https://img.shields.io/badge/PyYAML-CB171E?logo=yaml&logoColor=white)
  
## 4. Dataset

- Real Image: 500장
- Fake Image: 500장
- Total: 1,000장
- Train : Validation = 8:2

### 4-1 Dataset Structure

```text
data/
├── train/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

> **Dataset Download**
>
> GitHub 용량 제한으로 인해 데이터셋은 Google Drive를 통해 제공합니다.
>
> 🔗 [Google Drive Link](여기에 링크)
>
> 포함 항목
> 
> - Train Dataset
>   
> - Validation Dataset
>   
> - Image Metadata (Excel)


## 5. Model Architecture

본 프로젝트의 전체 딥페이크 탐지 파이프라인은 아래와 같다.
<p align="center">
<img width="440" height="310" alt="FlowChart" src="https://github.com/user-attachments/assets/749efe42-5e54-45cf-aae1-15e17462f8d7" />
</p>
<p align="center">
  <i>Overall Deepfake Detection Flowchart</i>
</p>

본 프로젝트에서는 다음 딥러닝 모델들의 성능을 비교하였다.

- EfficientNet-B4
- MobileNetV3
- ResNet-50
- ViT-B/16

또한 RGB 이미지와 Artifact Map을 결합한 앙상블 모델을 구성하여 최종 성능을 비교하였다.

## 6. Experimental Results

- RGB 기반 모델 성능 평가
- Artifact 기반 모델 성능 평가
- RGB + Artifact 앙상블 기반 모델 성능 평가
- 앙상블 모델 성능 비교
- 입력 해상도에 따른 성능 비교
- Confusion Matrix 분석

### 6-1. 모델별 성능 비교
<p>
<img width="347" height="277" alt="모델별 비교" src="https://github.com/user-attachments/assets/eb9893c3-d029-478e-ba02-e46a2edf2b52" />
</p>
입력 데이터 유형에 따라 최적의 모델이 달랐으며, RGB 기반에서는 MobileNetV3, Artifact 기반에서는 EfficientNet-B4가 가장 우수한 성능을 보였다. 최종적으로 RGB와 Artifact를 결합한 Ensemble에서는 EfficientNet-B4 기반 모델이 가장 높은 탐지 성능을 나타냈다.

### 6-2. 기본 입력 해상도(224×224) 성능 비교
<p>
<img width="517" height="250" alt="EN224ver_최종" src="https://github.com/user-attachments/assets/1ca6ef4d-a894-4c32-bf0b-a05629915af5" />
</p>
모든 모델을 동일한 입력 해상도(224×224)에서 비교하여 기본 성능을 평가하였다.

### 6-3. EfficientNet 권장 입력 해상도 비교

EfficientNet은 모델별 권장 입력 해상도가 다르므로, 기본 입력 해상도(224×224)와 권장 입력 해상도(B4: 380×380, B5: 456×456)를 적용하여 성능 변화를 비교하였다.

| Model | Resolution | Accuracy |
|:------|:----------:|---------:|
| EfficientNet-B4 | 224 × 224 | 67.70% |
| EfficientNet-B4 | **380 × 380** | **85.90%** |
| EfficientNet-B5 | 224 × 224 | 57.70% |
| EfficientNet-B5 | **456 × 456** | **71.40%** |

> 권장 입력 해상도를 적용한 결과 두 모델 모두 성능이 향상되었으며, 특히 EfficientNet-B4에서 가장 큰 성능 향상을 확인하였다. 반면 EfficientNet-B5는 성능이 향상되었음에도 B4보다 낮은 정확도를 보였으며, 이는 데이터셋 규모, 학습 조건 및 모델 특성 등의 영향으로 판단되며, 추가적인 데이터 확장과 하이퍼파라미터 최적화를 통해 개선될 가능성이 있다.

### 6-4. Confusion Matrix
<p>
<img width="954" height="703" alt="Matrix" src="https://github.com/user-attachments/assets/26b65419-4a89-45a6-a2b3-5c6bc5837408" />
</p>
Confusion Matrix를 통해 Real/Fake 분류 성능과 주요 오분류 패턴을 분석하였다.

## 7. Web Application

학습된 모델을 활용하여 Streamlit 기반의 딥페이크 탐지 웹 서비스를 구현하였다.

- 이미지 업로드를 통한 딥페이크 탐지
- JPEG 압축률 조절 기능 제공
- 압축률 변화에 따른 모델의 탐지 성능 및 강건성(Robustness) 확인
- Real/Fake 예측 결과 출력

※ 현재 웹 환경과 학습 환경 간 추론 결과 차이를 개선하기 위한 최적화를 진행 중이다.
[이미지 업로드 화면 넣어라 까먹지말고 나현아]

### 7-1. Demo
🔗 Streamlit Web Application
(링크)

※ 현재 성능 최적화를 진행 중입니다.

## 8. Future Work
- 웹 환경에서의 추론 성능 개선
- 다양한 데이터셋을 활용한 일반화 성능 향상
- Vision Transformer 기반 모델 성능 개선

## 9. Contributors

- 김나현
- 박민혁 (Team Leader)
- 윤혁준

## 10. My Contributions

- 딥페이크 학습 데이터 생성
- 이미지 생성 프롬프트 및 메타데이터(Excel) 구축
- 오분류 분석을 위한 데이터셋 관리
- 최신 베이스라인 모델 조사 및 학습 파이프라인 구축
- 프로젝트 아키텍처 및 플로우차트 설계
- Streamlit 기반 딥페이크 탐지 웹 애플리케이션 구현
- JPEG 압축률 조절 기능 구현 및 강건성 평가
- 모델 성능 평가 및 결과 시각화
- 최종 발표 자료(PPT) 공동 제작


