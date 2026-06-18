# <h1 align="center">CNN-based Deepfake Detection</h1>

<p align="center">
RGB 및 Artifact 기반 딥페이크 탐지 모델 개발 프로젝트
</p>

## Project Structure

src/                학습 및 추론 코드

configs/            모델 설정 파일

models/             학습된 모델

results/            실험 결과

dataset/            데이터셋 안내

## Project Overview

본 프로젝트는 CNN 기반 딥러닝 모델을 활용하여 이미지의 딥페이크 여부를 탐지하는 것을 목표로 한다.

RGB 이미지와 Artifact Map을 각각 활용하여 탐지 성능을 비교하고, 두 정보를 결합한 앙상블 기법을 적용하여 탐지 성능을 향상시키고자 하였다.


## Features
- RGB 이미지 기반 딥페이크 탐지
- Artifact Map 기반 딥페이크 탐지
- RGB + Artifact 앙상블 모델 성능 비교
- EfficientNet-B4, MobileNetV3, ResNet-50, ViT-B/16 성능 비교
- 입력 해상도에 따른 모델 성능 비교
- Threshold 및 Confusion Matrix 분석
- Streamlit 기반 딥페이크 탐지 웹 서비스 구현
- Vision Transformer(16x16 patch) 기반 추가 실험 수행


## Dataset

- Real Image: 500장
- Fake Image: 500장
- Total: 1,000장
- Train : Validation = 8 : 2


## Model Architecture

본 프로젝트에서는 다음 모델들의 성능을 비교하였다

- EfficientNet-B4
- MobileNetV3
- ResNet-50
- ViT-B/16

또한 RGB 이미지와 Artifact Map을 활용한 앙상블 모델을 구성하여 성능을 비교하였다.


## Results

- RGB 기반 모델 성능 평가
- Artifact 기반 모델 성능 평가
- 앙상블 모델 성능 비교
- 입력 해상도에 따른 성능 비교
- Confusion Matrix 분석


## Web Application

학습된 모델을 활용하여 Streamlit 기반의 딥페이크 탐지 웹 서비스를 구현하였다.

- 이미지 업로드를 통한 딥페이크 탐지
- JPEG 압축률 조절 기능 제공
- 압축률 변화에 따른 모델의 탐지 성능 및 강건성(Robustness) 확인
- Real/Fake 예측 결과 출력

※ 현재는 웹 환경과 학습 환경의 차이로 인해 추론 성능 개선을 진행 중이다.


## Future Work
- 웹 환경에서의 추론 성능 개선
- 다양한 데이터셋을 활용한 일반화 성능 향상
- Vision Transformer 기반 모델 성능 개선


## Contributors

### 김나현
- 딥페이크 이미지 생성(150개)
- 이미지 excel 정리
- streamlit 기반 사이트 구현
- 모델 성능 평가 및 결과 분석

### 박민혁(Team Leader)
- 메인 코드 구축 및 프로젝트 총괄
- 딥페이크 이미지 생성(150개)
- 타 모델과의 성능 비교 분석
- 모델 성능 평가 및 결과 분석

### 윤혁준
- 딥페이크 이미지 생성(200개)
- 보고서 제작
- 타 모델과의 성능 비교 분석
- 모델 성능 평가 및 결과 분석
