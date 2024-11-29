import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # reward 값들을 클래스 변수로 정의
        self.SURVIVAL_REWARD = 0.1    # 생존 보상 (기본: 0.1)
        self.PIPE_REWARD = 1.0         # 파이프 통과 보상 (기본: 1.0)
        self.COLLISION_PENALTY = -1.0   # 충돌 패널티 (기본: -1.0)
        self.JUMP_REWARD = 0.3         # 점프 보상 (새로 추가)
        self.prev_score = 0            # prev_score 초기화 추가

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 기본 reward를 생존 보상으로 설정
        reward = self.SURVIVAL_REWARD
        
        # 점프(action=1)할 때 추가 보상
        if action == 1:  # 1이 점프 액션
            reward += self.JUMP_REWARD
        
        # 파이프 통과 시 보상
        if info.get('score', 0) > self.prev_score:
            reward = self.PIPE_REWARD
            self.prev_score = info['score']
        
        # 충돌 시 패널티
        if terminated:
            reward = self.COLLISION_PENALTY
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_score = 0  # reset에서도 prev_score 초기화
        return super().reset(**kwargs)
