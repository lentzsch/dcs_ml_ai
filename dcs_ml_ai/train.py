# Entry point for training the DCS ML agent
from stable_baselines3 import PPO
from dcs_ml_ai.envs.wrappers import BasicEnvWrapper

def main():
    env = BasicEnvWrapper("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("ppo_cartpole")

if __name__ == "__main__":
    main()
