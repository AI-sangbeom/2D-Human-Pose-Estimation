# [Review] 2D Human Pose Estimation: A Survey
H Chen, R Feng, S Wu, H Xu, F Zhou, Z Liu - Multimedia systems, 2023 - Springer

> **Keywords** : Human pose estimation, pose estimation survey, deeplearning, convolutional neural network



## 1. Introduction

<div>
<img width="700" src='./images/human pose estimation on multimedia data.png'></img>
</div>

#### The Goal of HPE


1. 다양한 데이터 환경에서 사람을 찾는 것

2. 각 사람들에게서 keypoints를 찾아내는 것 

        
#### Three Categories of Human Pose Estimation

1. Network architecture design

2. Network training refinement 

3. Post processing 

<br>

## 2. Problem Statement
**HPE 목표** : ***입력된 이미지 내의 모든 사람들의 포즈를 추론하는 것***

#### Challenges
1. 실제 환경에서는 과·소노출이나 사람-사물 얽힘과 같은 잡음 현상이 자주 발생하며, 이는 쉽게 탐지 실패로 이어질 수 있다.

2. 인간의 운동 사슬은 매우 유연하기 때문에, 많은 상황에서 자세 가림(occlusion)이나 자기 가림(self-occlusion)이 불가피하며, 이는 시각적 특징을 사용하는 키포인트 검출기를 더욱 혼란스럽게 만든다.

3. 모션 블러나 초점 흐림(defocus)은 영상에서 자주 나타나며, 이는 자세 추정의 정확도를 저하시킨다.

<br>

## 3. Network Architecture Design Methods

Pose Estimation 네트워크 구조는 크게 2개 (**top-down, bottom-up**) 로 나뉜다. 

#### Top-Down 
<img src='./images/TopDownParadigm.png' width=700></img>

    1. 사람의 bounding box를 먼저 탐지
    2. 각 bounding box에 대해 단일 인물의 자세 추정을 수행함 

- **Top-Down Approaches**
  
    - regression-based
    - heatmap-based
    - video-based
    - model compressing-based

#### Bottom-Up

    1. 이미지 내에서 모든 키포인트들을 검출
    2. 이후, 서로 다른 사람의 인스턴스로 그룹화 진행

- **Bottom-Up Approaches**
  
    - one stage
    - two stage

<br>

### 3.1 Top-Down Framework
### 3.1.1 Regression-Based Methods

회귀 기반(regression-based) 접근법은 입력 이미지에서 인간의 운동학적 관절 좌표를 직접 회귀(regress) 하는 방식으로, 초기 연구들은 end-to-end 네트워크를 통해 이미지에서 키포인트 좌표를 바로 예측하는 모델을 제안함.

#### 초기 모델과 발전

- DeepPose [161]

    - AlexNet 기반 CNN으로 이미지 특징을 반복적으로 추출한 뒤 FC layer로 관절 좌표를 회귀.

    - 이 연구를 계기로 기존 전통 기법에서 딥러닝 기반 방식으로 전환이 가속됨.

#### 후속 연구

- Human pose estimation with iterative error feedback.

    - GoogleNet 기반 self-correcting 모델: 관절 위치를 직접 예측하지 않고 점진적으로 수정하는 방식 제안.

- Compositional human pose regression

    - 구조 정보를 활용한 regression 방식: 뼈(bone) 기반 재파라미터화된 포즈 표현을 사용하여 구조적 정보를 더 잘 반영.

- Semi-supervised classification with graph convolutional networks

    - GCN 기반 접근

    - 인체를 그래프로 모델링(노드=관절, 엣지=뼈)
    
    - 이미지 가이드를 활용해 보이지 않는 관절도 추정하는 Progressive GCN 모듈 제안.

- Pose recognition with cascade transformers

     - Transformer 기반 접근

     - Cascaded Transformer 구조로 사람 검출 → 각 인물의 관절 좌표 회귀를 end-to-end로 수행.

#### 장점과 한계

- 장점

    - **연산 효율이 높고, 실시간(real-time) 응용 가능성이 큼**.

- 한계

    - 각 관절을 단일 2D 좌표로 직접 출력하기 때문에 부위 영역(body part area)에 대한 **불확실성을 고려하지 못함**.

- 이를 보완하기 위해, 이후에는 **확률적 히트맵 기반(heatmap-based)** 방식이 도입되어 키포인트를 픽셀 단위 확률 분포로 예측하는 방법이 등장함.

<br>

### 3.1.2 Heatmap-Based Methods
