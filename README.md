# 🐦 Flappy Bird DQN Project

> Deep Q-Network (DQN)을 활용한 Flappy Bird 강화학습 프로젝트

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Flappy%20Bird-green.svg)](https://gymnasium.farama.org/)

## 📖 프로젝트 소개

Flappy Bird 게임을 DQN(Deep Q-Network) 알고리즘으로 학습시켜 자동으로 플레이하는 AI 에이전트를 개발하는 프로젝트입니다.

### 게임 규칙
- **터치/클릭**: 새가 위로 날아오름 (FLY)
- **아무것도 안함**: 새가 아래로 떨어짐 (FALL)
- **목표**: 파이프를 통과하며 최대한 오래 살아남기

---

## 🎯 프로젝트 목표 (Project Aims)

| # | 실험 항목 | 설명 |
|---|----------|------|
| 1 | **학습량 비교** | Episode 수(10K, 100K, 500K)에 따른 학습 효율 비교 |
| 2 | **Discount Factor** | γ 값(0.99 vs 0.90) 조정에 따른 성능 비교 |
| 3 | **Hidden Layer** | Hidden layer 개수(1개 vs 2개)에 따른 비교 |
| 4 | **Naïve DQN** | Naïve DQN과 DQN의 성능 비교 |
| 5 | **Custom Environment** | 커스텀 환경에서의 학습 실험 |

---

## 🧠 DQN 구조

### State (상태 공간) - 12차원

| # | State | 설명 |
|---|-------|------|
| 1-3 | Last Pipe | 마지막 파이프의 x, 상단 y, 하단 y 좌표 |
| 4-6 | Next Pipe | 다음 파이프의 x, 상단 y, 하단 y 좌표 |
| 7-9 | Next Next Pipe | 그 다음 파이프의 x, 상단 y, 하단 y 좌표 |
| 10 | Player Y | 플레이어(새)의 y 좌표 |
| 11 | Player Velocity | 플레이어의 y 방향 속도 |
| 12 | Player Rotation | 플레이어의 회전 각도 |

### Action (행동 공간)
- `0`: Nothing (아무것도 안함)
- `1`: Jump (점프)

### Reward (보상 체계)

| 상황 | 보상 |
|------|------|
| 생존 (매 프레임) | +0.1 |
| 파이프 통과 | +1.0 |
| 사망 | -1.0 |
| 화면 상단 충돌 | -0.5 |

### Network Architecture

```
Input (State: 12) → Hidden Layer (512) → Output (Actions: 2)
                         ↓
                  [Replay Buffer]
                         ↓
                  [Target Network]
```

---

## 📊 실험 결과

### 1️⃣ 학습량에 따른 성능 비교

| Episodes | Max Reward | Time |
|----------|------------|------|
| 10,000 | 6.0 | 5분 |
| 100,000 | 50.0 | 46분 |
| 500,000 | 80.0 | 3시간 |

### 2️⃣ Discount Factor 비교 (100K Episodes)

| Discount Factor | Best Reward | Time |
|-----------------|-------------|------|
| γ = 0.99 | 59.9 | 46분 |
| **γ = 0.90** | **82.9** | 1시간 42분 |

> 💡 **인사이트**: 즉각적인 파이프 통과에 집중해야 하므로, 낮은 discount factor가 더 효과적!

### 3️⃣ Hidden Layer 개수 비교

| Hidden Layers | Episodes | Best Reward | Time |
|---------------|----------|-------------|------|
| 1개 | 500,000 | 80.6 | 3시간 |
| **2개** | 250,000 | **100.1** | 1시간 36분 |

> 💡 **인사이트**: Hidden layer 추가로 네트워크 표현력이 증가하여 더 적은 에피소드로 높은 성능 달성!

### 4️⃣ Naïve DQN vs DQN (100K Episodes)

| 방식 | Best Reward | Time |
|------|-------------|------|
| Original DQN | 59.9 | 46분 |
| **γ=0.90 DQN** | **82.9** | 1시간 42분 |
| Naïve DQN | 50.6 | 54분 |

### 5️⃣ Custom Environment 실험

| 설정 | Episodes | Best Reward | Time |
|------|----------|-------------|------|
| Custom Env | 500,000 | 71.8 | 18시간 30분 |

### 🔀 최적 설정 조합 실험

| 설정 | Episodes | Best Reward | Time |
|------|----------|-------------|------|
| Hidden 2 + γ=0.90 | 300,000 | 88.5 | 1시간 57분 |

> ⚠️ **Discussion**: Hidden layer 2개와 Discount factor 0.90을 결합했을 때, 복잡한 네트워크가 단기적 보상에 **과적합(Overfitting)** 될 수 있음

---

## 📁 프로젝트 구조

```
FlappyBird/
├── FlappyBird1/              # 기본 DQN 구현 (학습량 비교)
├── FlappyBird2/              # Discount Factor 실험
├── FlappyBird3/              # Custom Environment 실험
├── FlappyBird_HiddenLayer/   # Hidden Layer 개수 비교
├── FlappyBird_NaiveDqn/      # Naïve DQN 구현
└── README.md
```

### 각 폴더 공통 구조

```
FlappyBird_X/
├── agent.py              # 학습 에이전트 (메인 학습 로직)
├── dqn.py                # DQN 신경망 구조
├── experience_replay.py  # Replay Buffer 구현
├── hyperparameters.yml   # 하이퍼파라미터 설정
├── play.py               # 학습된 모델 테스트
└── runs/                 # 학습 결과 (모델, 로그, 그래프)
```

---

## ⚙️ Hyperparameters

```yaml
replay_memory_size: 100000
mini_batch_size: 32
epsilon_init: 1.0
epsilon_decay: 0.99995
epsilon_min: 0.05
network_sync_rate: 10
learning_rate: 0.0001
discount_factor: 0.99  # 또는 0.90
fc1_nodes: 512
```

---

## 🚀 사용 방법

### 설치

```bash
pip install gymnasium
pip install flappy-bird-gymnasium
pip install torch
pip install matplotlib
pip install pyyaml
```

### 학습

```bash
cd FlappyBird2
python agent.py flappybird1 --train --max_episodes 100000
```

### 테스트 (학습된 모델 실행)

```bash
python agent.py flappybird1
```

---

## 🔧 핵심 알고리즘

### Epsilon-Greedy Policy
- 탐험(Exploration)과 활용(Exploitation)의 균형
- ε 값이 점진적으로 감소 (1.0 → 0.05)

### Experience Replay
- 과거 경험을 버퍼에 저장
- 무작위 샘플링으로 상관관계 제거

### Target Network
- 안정적인 학습을 위한 별도의 타겟 네트워크
- 주기적으로 policy network의 가중치 복사

---

## 📈 향후 개선 방향

- [ ] Python 버전 최적화
- [ ] Custom Environment 심화 이해
- [ ] Mean 값 추적 기능 강화
- [ ] Grid Search를 통한 하이퍼파라미터 최적화
- [ ] Dueling DQN, Prioritized Experience Replay 적용

---

## 👥 팀원

- **김정찬**
- **홍유빈**

---

## 📚 참고 자료

- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

---

## 📄 License

This project is for educational purposes - SMU 2024-2 강화학습 프로젝트

