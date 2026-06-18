# <h1 align="center">CNN 기반 딥페이크 탐지 모델</h1>
# (CNN-based Deepfake Detection)

## 프로젝트 소개
본 프로젝트는 CNN 기반 딥러닝 모델을 활용하여 이미지의 딥페이크 여부를 탐지하는 것을 목표로 한다.

RGB 이미지와 Artifact Map을 각각 입력으로 사용하는 모델의 성능을 비교하고, 두 정보를 결합한 앙상블 기법을 적용하여 탐지 성능을 분석하였다.

## 주요 내용
RGB 이미지 기반 딥페이크 탐지
Artifact Map 기반 딥페이크 탐지
RGB + Artifact 앙상블 모델 성능 비교
EfficientNet-B4, MobileNetV3, ResNet-50, ViT-B/16 성능 비교
입력 해상도에 따른 모델 성능 비교
Threshold 및 Confusion Matrix 분석
Streamlit 기반 딥페이크 탐지 웹 서비스 구현

## 데이터셋
