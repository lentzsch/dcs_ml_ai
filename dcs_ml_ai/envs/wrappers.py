 
import gymnasium as gym

class BasicEnvWrapper(gym.Wrapper):
    def __init__(self, env_name: str, **kwargs):
        env = gym.make(env_name, **kwargs)
        super().__init__(env)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info
