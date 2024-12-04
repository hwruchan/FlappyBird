# FlappyBird DQN 프로젝트
By: 김정찬, 홍유빈

## 프로젝트 목표

이 프로젝트에서는 DQN(Deep Q-Learning Network)을 사용하여 Flappy Bird를 효율적으로 플레이하도록 AI 에이전트를 교육하는 방법을 살펴봅니다. 주요 목표는 다음과 같습니다.

1. 다양한 에피소드 수에 따른 학습 효율성을 비교합니다.
2. 숨겨진 레이어 수(1 대 2)의 영향을 평가합니다.
3. 할인 요인과 같은 하이퍼파라미터의 영향을 테스트합니다.
4. 향상된 DQN과 Naïve DQN의 성능을 비교합니다.
5. 사용자 정의 환경을 실험합니다.

## 게임 개요

게임 플레이: 새를 날리려면 탭하거나 클릭하세요. 아무것도 하지 않으면 넘어진다. 파이프를 통과하면서 최대한 오래 살아남으세요.

주 대표(체육관):
  마지막 및 다음 파이프 위치(x, y)와 다음 파이프 위치.
 플레이어의 수직 위치, 속도 및 회전.

작업:
  점프하거나 아무것도 하지 마세요.

보상 구조:
  파이프 통과 : +1
  죽어가는 중: -1.0
  화면 상단 터치: -0.5
  살아남기(프레임당): +0.1

## 주요 구성요소

1. 리플레이를 경험해보세요
저장된 경험에서 무작위로 미니 배치를 샘플링하여 연속적인 경험 간의 상관 관계를 줄이고 다양한 훈련 세트를 보장합니다.

2. 대상 네트워크
보조 네트워크는 잠시 동안 목표 예측을 수정하여 학습을 안정화하기 위해 주기적으로 업데이트됩니다.

3. 학습 알고리즘
손실 함수: 평균 제곱 오차(MSE).
훈련 단계:
  zero_grad(): 이전 그라데이션을 지웁니다.
  backward(): 손실을 기준으로 기울기를 계산합니다.
  step(): 계산된 기울기를 사용하여 가중치를 업데이트합니다.
실험 및 결과
1. 학습량
에피소드	최대 보상	

10,000  	6.0	     

100,000	  50	

500,000	  80

2. Amount of learning rate
할인율	에피소드	최고의 보상

0.99	  100,000	  59.9	

0.90	  100,000	  82.9	

4. Hidden Layer
숨겨진 레이어	에피소드	최고의 보상	

1	            500,000	  80.6	

2	            250,000	  100.1	

5. Naive DQN 대 DQN
모델	에피소드	최고의 보상	소요된 시간
원본 DQN	100,000	59.9	46분
순진한 DQN	100,000	50.6	54m
6. 맞춤형 환경
구성	에피소드	최고의 보상	소요된 시간
맞춤형 플래피버드	500,000	71.8	18시 30분
논의
2개의 히든 레이어와 할인 계수 0.90을 결합하면 성능이 향상되지만 단기 보상에 과적합될 위험이 있습니다.
최적화된 네트워크는 복잡한 패턴을 효과적으로 학습할 수 있지만 단기 보상과 장기 보상의 균형을 맞추려면 세심한 조정이 필요합니다.
추가 개선 사항에는 그리드 검색, 사용자 정의 환경 및 고급 초매개변수 조정이 포함됩니다.
