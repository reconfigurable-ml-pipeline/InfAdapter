import os

import ray
from ray.tune import tune
from ray.rllib.agents.ppo import PPOTrainer

from auto_tuner.simulation.config import get_environment_config
from auto_tuner.simulation.kube_cluster import SimulatedCluster

if __name__ == "__main__":
    ray.init()

    stop = {
        # "training_iteration": 50,
        "timesteps_total": 2500,
        # "episode_reward_mean": 60.0,
    }

    config = {
        "env": SimulatedCluster,
        # "framework": "torch",
        "env_config": get_environment_config()
    }
    analysis = tune.run(
        PPOTrainer, config=config, stop=stop, metric="episode_reward_mean", mode="max", name="rl_autoscaler",
        checkpoint_at_end=True, verbose=0
    )
    print("checkpoint path is:", analysis.best_checkpoint)
    with open(os.path.dirname(__file__) + "/checkpoint.txt", "w") as f:
        f.write(analysis.best_checkpoint)

    ray.shutdown()
