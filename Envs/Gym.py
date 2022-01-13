import os
import gym  # not necessary
import numpy as np
from copy import deepcopy

gym.logger.set_level(40)  # Block warning
def build_env(env, if_print=False, env_num=1, device_id=None, args=None, ):#####Used
    if isinstance(env, str):
        env_name = env
    else:
        env_name = env.env_name
        original_env = env
    env = None
    if env is None:
        try:
            env = deepcopy(original_env)
            print(f"| build_env(): Warning. NOT suggest to use `deepcopy(env)`. env_name: {env_name}")
        except Exception as error:
            print(f"| build_env(): Error. {error}")
            raise ValueError("| build_env(): register your custom env in this function.")
    return env
def build_eval_env(eval_env, env, env_num, eval_gpu_id, args, ):
    if isinstance(eval_env, str):
        eval_env = build_env(env=eval_env, if_print=False, env_num=env_num, device_id=eval_gpu_id, args=args, )
    elif eval_env is None:
        eval_env = build_env(env=env, if_print=False, env_num=env_num, device_id=eval_gpu_id, args=args, )
    else:
        assert hasattr(eval_env, 'reset')
        assert hasattr(eval_env, 'step')
    return eval_env