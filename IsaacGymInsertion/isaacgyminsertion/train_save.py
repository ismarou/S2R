# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging

from datetime import datetime
import os

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf

# noinspection PyUnresolvedReferences
import isaacgym

import hydra
from hydra.utils import to_absolute_path

from isaacgyminsertion.utils.rlgames_utils import multi_gpu_get_rank
from isaacgyminsertion.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from isaacgyminsertion.tasks import isaacgym_task_map
#from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
#from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
#from isaacgymenvs.tasks import isaacgym_task_map

import gym
import subprocess
import argparse
from typing import Optional
from termcolor import cprint

from isaacgyminsertion.utils.reformat import omegaconf_to_dict, print_dict
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed
#from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
#from isaacgymenvs.utils.utils import set_np_formatting, set_seed


from isaacgyminsertion.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
from isaacgyminsertion.utils.wandb_utils import WandbAlgoObserver
#from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
#from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from isaacgyminsertion.learning import amp_continuous
from isaacgyminsertion.learning import amp_players
from isaacgyminsertion.learning import amp_models
from isaacgyminsertion.learning import amp_network_builder
import isaacgyminsertion

#from isaacgymenvs.learning import amp_continuous
#from isaacgymenvs.learning import amp_players
#from isaacgymenvs.learning import amp_models
#from isaacgymenvs.learning import amp_network_builder
#import isaacgymenvs


#from algo.models.transformer.frozen_ppo import PPO # TODO: we can completely switch to this one for PPO. (added functions for online testing of offline training)
##from algo.ppo.ppo import PPO
#from algo.ext_adapt.ext_adapt import ExtrinsicAdapt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import wandb



def preprocess_train_config(cfg, config_dict):

    """
        
        Adding common configuration parameters to the rl_games train config.
        An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same variable interpolations in each config.

    """

    # Print all the key - value pairs in the config_dict
    print_dict(config_dict)

    train_cfg = config_dict['params']['config']
    #train_cfg = {}

    train_cfg['device'] = "cuda:0" #cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict



#@hydra.main(config_name="config", config_path="./cfg") # From isaacgyminsertion/train.py
@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #print(cfg.wandb_name)

    #run_name = f"{cfg.wandb_name}_{time_str}"
    #run_name = f"{cfg.train.params.config.name}_{time_str}"

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # Debugging: Print the full configuration
    print(OmegaConf.to_yaml(cfg))

    #cfg_dict = omegaconf_to_dict(cfg)
    #print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    '''

    # global rank of the GPU
    if cfg.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f"cuda:{rank}"
        cfg.rl_device = f"cuda:{rank}"
        cfg.graphics_device_id = int(rank)
        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(cfg.seed + rank)
    else:
        rank = -1
        cfg.seed = set_seed(cfg.seed)

    '''

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    # sets seed. if seed is -1 will pick a random one
    # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    # in trainV2.py it is commented out


    '''

    # From trainV2.py

    # for training the transformer with offline data only
    if cfg.offline_training:
        from algo.models.transformer.runner import Runner as TransformerRunner 
        from algo.models.transformer.frozen_ppo import PPO

        agent = None
        
        # perform train
        runner = TransformerRunner(cfg.offline_train, agent=agent)
        runner.run()
        
        exit()

    '''

    def create_isaacgym_env(**kwargs):
        
        envs = isaacgyminsertion.make(
                                            cfg.seed, 
                                            cfg.task_name, 
                                            cfg.task.env.numEnvs, 
                                            cfg.sim_device,
                                            cfg.rl_device,
                                            cfg.graphics_device_id,
                                            cfg.headless,
                                            cfg.multi_gpu,
                                            cfg.capture_video,
                                            cfg.force_render,
                                            cfg,
                                            **kwargs,
                                                                                )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                                                envs,
                                                f"videos/Ismarou_{time_str}",
                                                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                                                video_length=cfg.capture_video_len,
                                                                                                                        )
        return envs

    env_configurations.register('rlgpu', {
                                                'vecenv_type': 'RLGPU',
                                                'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
                                                                                                                        }
                                                                                                                                )
    
    '''

    # From trainV2.py

    envs = isaacgym_task_map[cfg.task_name](
                                                cfg = omegaconf_to_dict(cfg.task),
                                                sim_device = cfg.sim_device,
                                                rl_device = cfg.rl_device,
                                                graphics_device_id = cfg.graphics_device_id,
                                                headless = cfg.headless,
                                                virtual_screen_capture = False,  # cfg.capture_video,
                                                force_render = cfg.force_render  # if not cfg.headless else False,
                                                                                                                            )

    '''


    '''
    # From trainV2.py
    
    output_dif = os.path.join('outputs', str(datetime.now().strftime("%m-%d-%y")))
    output_dif = os.path.join(output_dif, str(datetime.now().strftime("%H-%M-%S")))
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(cfg.train.algo)(envs, output_dif, full_config=cfg)
    '''

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {
                                'names': list(actor_net_cfg.inputs.keys()), 
                                'concat': not actor_net_cfg.name == "complex_net", 
                                'space_name': 'observation_space'
                                                                                                    }
        

        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {
                                        'names': list(critic_net_cfg.inputs.keys()), 
                                        'concat': not critic_net_cfg.name == "complex_net", 
                                        'space_name': 'state_space'
                                                                                                                    }
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # Initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # Register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # Convert CLI arguments into dictionary
    # Create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # Dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
                    'train': not cfg.test,
                    'play': cfg.test,
                    'checkpoint': cfg.checkpoint,
                    'sigma': cfg.sigma if cfg.sigma != '' else None
                                                                            })



    '''
    
    # From trainV2.py

    if cfg.test:
        assert cfg.train.load_path
        agent.restore_test(cfg.train.load_path)
        if not cfg.offline_training_w_env:
            num_success, total_trials = agent.test()
            print(f"Success rate: {num_success / total_trials}")
        else:
            # from algo.models.transformer.frozen_ppo import PPO as PPOv3
            # agent = PPOv3(envs, output_dif, full_config=cfg)
            agent.restore_test(cfg.train.load_path)
            agent.set_eval()

            from algo.models.transformer.runner import Runner as TransformerRunner 
            runner = TransformerRunner(cfg.offline_train, agent, action_regularization=cfg.offline_train.train.action_regularization)
            runner.run()

        # sim_timer = cfg.task.env.sim_timer
        # num_trials = 3
        # cprint(f"Running simulation for {num_trials} trials", "green", attrs=["bold"])
        # thread_stop = threading.Event()
        # agent.restore_test(cfg.train.load_path)
        # sim_thread = threading.Thread(
        #     name="agent.test()", target=agent.test, args=[thread_stop]
        # )
        # threading.Thread(
        #     name="sim_time", target=agent.play_games, args=[thread_stop, num_trials]
        # ).start()

        # sim_thread.start()
        # sim_thread.join()
        # cprint(f"Simulation terminated", "green", attrs=["bold"])
    else:
        if rank <= 0:
            date = str(datetime.now().strftime("%m%d%H"))
            with open(os.path.join(output_dif, f"config_{date}.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        # check whether execute train by mistake:
        # best_ckpt_path = os.path.join(
        #     "outputs",
        #     cfg.train.ppo.output_name,
        #     "stage1_nn" if cfg.train.algo == "PPO" else "stage2_nn",
        #     "best.pth",
        # )
        # user_input = 'yes'
        # if os.path.exists(best_ckpt_path):
        #     user_input = input(
        #         f"are you intentionally going to overwrite files in {cfg.train.ppo.output_name}, type yes to continue \n"
        #     )
        #     if user_input != "yes":
        #         exit()

        agent.restore_train(cfg.train.load_path)
        agent.train()

    cprint("Finished", "green", attrs=["bold"])
    
    '''



if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument("config_path", type=argparse.FileType("r"), help="Path to hydra config.")
    # args = parser.parse_args()

    launch_rlg_hydra()
