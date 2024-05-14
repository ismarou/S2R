# Copyright (c) 2021-2023, NVIDIA Corporation
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

"""
    Factory: Class for insertion task.

    Inherits insertion environment class and abstract task class (not enforced). Can be executed with
    python train.py task=FactoryTaskInsertionTactile

    Only the environment is provided; training a successful RL policy is an open research problem left to the user.
"""

import hydra
import omegaconf
import time
import os
import torch
import numpy as np
import math
import time

from isaacgym import gymapi, gymtorch
from isaacgyminsertion.tasks.factory_tactile.factory_env_insertion import FactoryEnvInsertionTactile
from isaacgyminsertion.tasks.factory_tactile.factory_schema_class_task import FactoryABCTask
from isaacgyminsertion.tasks.factory_tactile.factory_schema_config_task import FactorySchemaConfigTask
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from isaacgyminsertion.tasks.factory_tactile.factory_utils import *
from isaacgyminsertion.utils import torch_jit_utils
from multiprocessing import Process, Queue, Manager
import cv2
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
torch.set_printoptions(sci_mode=False)


class FactoryTaskInsertionTactile(FactoryEnvInsertionTactile, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        """
            Initialize instance variables. Initialize task superclass.
        """

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()

        self._acquire_task_tensors()
        self.parse_controller_spec()

        self.temp_ctr = 0
        # self.fig_for_plots = plt.figure()
        # self.axes = self.axes.flatten()

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='kuka_task_insertion')

        
    def _get_task_yaml_params(self):
        
        """
            Initialize instance variables from YAML files.
        """

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskInsertionTactilePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        
        """
            Acquire tensors.
        """

        self.identity_quat = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

        self.plug_grasp_pos_local = self.plug_heights * 0.95 * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.plug_tip_pos_local = self.plug_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self.socket_tip_pos_local = self.socket_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        
        # Compute pose of gripper goal and top of socket in socket frame
        #self.gripper_goal_pos_local = torch.tensor([[0.0,0.0, (self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]),] for i in range(self.num_envs)],device=self.device,)
        #self.gripper_goal_quat_local = self.identity_quat.clone()

        # Gripper pointing down w.r.t the world frame
        gripper_goal_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial, device=self.device).unsqueeze(0).repeat((self.num_envs, 1))
        self.gripper_goal_quat = torch_jit_utils.quat_from_euler_xyz(gripper_goal_euler[:, 0], gripper_goal_euler[:, 1], gripper_goal_euler[:, 2])

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_plug = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device, )
        self.keypoints_socket = torch.zeros_like(self.keypoints_plug, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.cfg_task.env.numTargets), dtype=torch.float, device = self.device)

        
        self.contact_points_hist = torch.zeros((self.num_envs, self.cfg_task.env.num_points * 1), dtype=torch.float, device=self.device)
        
        
        self.plug_socket_dist = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        
        # reset tensors
        self.timeout_reset_buf = torch.zeros_like(self.reset_buf)
        self.degrasp_buf = torch.zeros_like(self.reset_buf)
        self.far_from_goal_buf = torch.zeros_like(self.reset_buf)
        self.success_reset_buf = torch.zeros_like(self.reset_buf)

        # state tensors
        self.plug_hand_pos, self.plug_hand_quat = torch.zeros((self.num_envs, 3), device=self.device), torch.zeros((self.num_envs, 4), device=self.device)
        self.rigid_physics_params = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float)  # TODO: Take num_params to config
        self.finger_normalized_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.gt_extrinsic_contact = torch.zeros((self.num_envs, self.cfg_task.env.num_points), device=self.device, dtype=torch.float)

        # reward tensor
        self.reward_log_buf = torch.zeros_like(self.rew_buf)

        self.ep_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ep_success_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ep_failure_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.for_plots = {}

        self.dec = torch.zeros((self.num_envs, self.cfg_task.env.num_points), device=self.device, dtype=torch.float)

    def _refresh_task_tensors(self):
        
        """
            Refresh tensors.
        """

        self.refresh_base_tensors()
        self.refresh_env_tensors()

        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(
                                                                                    self.plug_quat,
                                                                                    self.plug_pos,
                                                                                    self.plug_grasp_quat_local,
                                                                                    self.plug_grasp_pos_local
                                                                                                                    )
        

        # Add observation noise to plug pos
        
        #self.noisy_plug_pos = torch.zeros_like(self.plug_pos, dtype=torch.float32, device=self.device)
        #plug_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)- 0.5)
        #plug_obs_pos_noise = plug_obs_pos_noise @ torch.diag(torch.tensor(self.plug_pos_obs_noise,dtype=torch.float32,device=self.device,))
        #self.noisy_plug_pos[:, 0] = self.plug_pos[:, 0] + plug_obs_pos_noise[:, 0]
        #self.noisy_plug_pos[:, 1] = self.plug_pos[:, 1] + plug_obs_pos_noise[:, 1]
        #self.noisy_plug_pos[:, 2] = self.plug_pos[:, 2] + plug_obs_pos_noise[:, 2]
        self.noisy_plug_pos = self.plug_pos.clone()

        # Add observation noise to plug rot
        
        #plug_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        #plug_obs_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
        #plug_obs_rot_noise = plug_obs_rot_noise @ torch.diag(torch.tensor(self.cfg_task.env.plug_rot_obs_noise, dtype=torch.float32, device=self.device,))
        #plug_obs_rot_euler = plug_rot_euler + plug_obs_rot_noise
        #self.noisy_plug_quat = torch_jit_utils.quat_from_euler_xyz(plug_obs_rot_euler[:, 0], plug_obs_rot_euler[:, 1], plug_obs_rot_euler[:, 2])
        self.noisy_plug_quat = self.plug_quat.clone() 

        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(self.socket_pos, dtype=torch.float32, device=self.device)
        socket_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(torch.tensor(self.cfg_task.env.socket_pos_obs_noise, dtype=torch.float32, device=self.device,))

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        socket_obs_rot_noise = 2 * ( torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(torch.tensor( self.cfg_task.env.socket_rot_obs_noise, dtype=torch.float32, device=self.device,))

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_jit_utils.quat_from_euler_xyz(socket_obs_rot_euler[:, 0], socket_obs_rot_euler[:, 1], socket_obs_rot_euler[:, 2])

        # Compute observation noise on socket
        ( self.noisy_gripper_goal_quat, self.noisy_gripper_goal_pos,) = torch_jit_utils.tf_combine(
                                                                                                        self.noisy_socket_quat,
                                                                                                        self.noisy_socket_pos,
                                                                                                        self.gripper_goal_quat,
                                                                                                        self.socket_tip_pos_local,
                                                                                                                                        )

        # Compute pos of keypoints on gripper, socket, and plug in world frame
        socket_tip_pos_local = self.socket_tip_pos_local.clone()
        socket_tip_pos_local[:, 2] -= self.socket_heights.view(-1)
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(
                                                                        self.plug_quat,
                                                                        self.plug_pos,
                                                                        self.identity_quat,
                                                                        (keypoint_offset * self.socket_heights)
                                                                                                                )[1]
            
            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(
                                                                            self.socket_quat,
                                                                            self.socket_pos,
                                                                            self.identity_quat,
                                                                            (keypoint_offset * self.socket_heights) + socket_tip_pos_local
                                                                                                                                                )[1]
            
        # Fingertip forces
        e = 0.9 if self.cfg_task.env.smooth_force else 0
        normalize_forces = lambda x: (torch.clamp(torch.norm(x, dim=-1), 0, 50) / 50).view(-1)

        # Normalize_forces = lambda x: (torch.norm(x, dim=-1)).view(-1)
        self.finger_normalized_forces[:, 0] = (1 - e) * normalize_forces(self.left_finger_force.clone()) + e * self.finger_normalized_forces[:, 0]
        self.finger_normalized_forces[:, 1] = (1 - e) * normalize_forces(self.right_finger_force.clone()) + e * self.finger_normalized_forces[:, 1]
        self.finger_normalized_forces[:, 2] = (1 - e) * normalize_forces(self.middle_finger_force.clone()) + e * self.finger_normalized_forces[:, 2]
        
        if "left_finger_pos" not in self.for_plots:
            self.for_plots["left_finger_pos"] = []
        if "right_finger_pos" not in self.for_plots:
            self.for_plots["right_finger_pos"] = []
        if "middle_finger_pos" not in self.for_plots:
            self.for_plots["middle_finger_pos"] = []

        left_finger_pos_wrt_eef, _ = self.pose_world_to_hand_base(self.left_finger_pos.clone(), self.left_finger_quat.clone(), as_matrix=False)
        self.for_plots["left_finger_pos"].append(left_finger_pos_wrt_eef[0, 0].item())
        self.for_plots["right_finger_pos"].append(left_finger_pos_wrt_eef[0, 1].item())
        self.for_plots["middle_finger_pos"].append(left_finger_pos_wrt_eef[0, 2].item())

    
    def pre_physics_step(self, actions):
        
        """
            Reset environments. 
            Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains.
        """

        self.prev_actions[:] = self.actions.clone()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        # self.actions[:, -1] = 0.0 # don't apply z-axis rotation
        # test actions for whenever we want to see some axis motion
        # self.actions[:, :] = 0.
        # self.actions[:, 2] = -1.0 # if (self.progress_buf[0].item() % 100) < 50 else -1.0

        delta_targets = torch.cat([
                                        self.actions[:, :3] @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)),  # 3
                                        self.actions[:, 3:6] @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))  # 3
                                                                                                                                                        ], dim=-1).clone()

        # Update targets
        self.targets = self.prev_targets + delta_targets

 
        self._apply_actions_as_ctrl_targets(actions = self.actions, ctrl_target_gripper_dof_pos = self.ctrl_target_gripper_dof_pos, do_scale = True)
        self.prev_targets[:] = self.targets.clone()


    def post_physics_step(self):
        
        """
            Step buffers. Refresh tensors. Compute observations and reward.
        """

        self.progress_buf[:] += 1
        self.randomize_buf[:] += 1
        # print('progress_buf', self.progress_buf[0])

        # In this policy, episode length is constant
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.temp_ctr += 1

        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

        # plots = ["plug_hand_pos_x", "plug_hand_pos_y", "plug_hand_pos_z"] # ["left_finger_forces", "right_finger_forces", "middle_finger_forces"]
        # plots = [] # ["left_finger_pos", "right_finger_pos", "middle_finger_pos"] # ["left_finger_forces", "right_finger_forces", "middle_finger_forces"]
        # colors = ['red', 'blue', 'green']
        # self.axes = []

        # if len(plots) > 0:
        #     if len(self.axes) == 0:
        #         for ax_cl_idx in range(len(plots)):
        #             self.axes.append(self.fig_for_plots.add_subplot(len(plots), 1, ax_cl_idx + 1))
        #     else:
        #         for ax_cl_idx in range(len(plots)):
        #             self.axes[ax_cl_idx].clear()
        #     plt_idx = 0
        #     for k, v in self.for_plots.items():
        #         if k in plots:
        #             if len(v) > 50:
        #                 v.pop(0)
        #             self.axes[plt_idx].plot(v, color=colors[plots.index(k)], label=k)
        #             plt_idx += 1
        #     plt.pause(0.0001)
        # else:
        #     plt.close(self.fig_for_plots)

        # for k, v in self.for_plots.items():
        #     self.for_plots[k] = v[-50:]

        if self.viewer or True:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            rotate_vec = lambda q, x: quat_apply(q, to_torch(x, device=self.device) * 0.2).cpu().numpy()
            num_envs = 1
            ref_lines = False
            if ref_lines:
                for i in range(num_envs):
                    actions = self.actions[i, :].clone().cpu().numpy()
                    keypoints = self.keypoints_plug[i].clone().cpu().numpy()
                    quat = self.plug_quat[i, :]

                    # for j in range(self.cfg_task.rl.num_keypoints):
                    #     ob = keypoints[j]
                    #     targetx = ob + rotate_vec(quat, [actions[0], 0, 0])
                    #     targety = ob + rotate_vec(quat, [0, actions[1], 0])
                    #     targetz = ob + rotate_vec(quat, [0, 0, actions[2]])

                    #     self.gym.add_lines(self.viewer, self.envs[i], 1,
                    #                        [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                    #     self.gym.add_lines(self.viewer, self.envs[i], 1,
                    #                        [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                    #     self.gym.add_lines(self.viewer, self.envs[i], 1,
                    #                        [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])
                    # print(keypoints)

                    for j in range(self.cfg_task.rl.num_keypoints):
                        ob = keypoints[j]
                        targetx = ob + rotate_vec(quat, [1, 0, 0])
                        targety = ob + rotate_vec(quat, [0, 1, 0])
                        targetz = ob + rotate_vec(quat, [0, 0, 1])

                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.85, 0.1, 0.1])
                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.85, 0.1, 0.1])

                for i in range(num_envs):
                    keypoints = self.keypoints_socket[i].clone().cpu().numpy()
                    quat = self.socket_quat[i, :]
                    # print(keypoints)

                    for j in range(self.cfg_task.rl.num_keypoints):
                        ob = keypoints[j]
                        targetx = ob + rotate_vec(quat, [1, 0, 0])
                        targety = ob + rotate_vec(quat, [0, 1, 0])
                        targetz = ob + rotate_vec(quat, [0, 0, 1])

                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targetx[0], targetx[1], targetx[2]], [0.1, 0.85, 0.1])
                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                        self.gym.add_lines(self.viewer, self.envs[i], 1, [ob[0], ob[1], ob[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.85, 0.1])

        self._render_headless()

    def compute_observations(self):
        
        """
            Compute observations.
        """

        self.gripper_goal_pos = self.socket_pos.clone()
        self.noisy_gripper_goal_pos = self.noisy_socket_pos.clone()

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        delta_quat = torch_jit_utils.quat_mul(self.gripper_goal_quat, torch_jit_utils.quat_conjugate(self.fingertip_centered_quat))
        noisy_delta_quat = torch_jit_utils.quat_mul(self.noisy_gripper_goal_quat, torch_jit_utils.quat_conjugate(self.fingertip_centered_quat))

        delta_plug_pos = self.socket_pos - self.plug_pos
        noisy_delta_plug_pos = self.noisy_socket_pos - self.noisy_plug_pos

        delta_plug_quat = torch_jit_utils.quat_mul(self.socket_quat, torch_jit_utils.quat_conjugate(self.plug_quat))
        noisy_delta_plug_quat = torch_jit_utils.quat_mul(self.noisy_socket_quat, torch_jit_utils.quat_conjugate(self.noisy_plug_quat))

        # Define observations (for actor)

        obs_tensors = [
                            self.pose_world_to_robot_base(self.noisy_plug_pos, self.noisy_plug_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.noisy_plug_pos, self.noisy_plug_quat)[1],  # 9
                            self.pose_world_to_robot_base(self.noisy_socket_pos, self.noisy_socket_quat)[0],  # 3
                            self.pose_world_to_robot_base(self.noisy_socket_pos, self.noisy_socket_quat)[1],  # 9
                                                                                                                                        ] # 24
        



        # Define state (for critic)
        '''
        state_tensors = [ 
                            self.arm_dof_pos,  # 7
                            self.arm_dof_vel,  # 7
                            self.pose_world_to_robot_base(  self.fingertip_centered_pos, self.fingertip_centered_quat )[0],  # 3
                            self.pose_world_to_robot_base(  self.fingertip_centered_pos, self.fingertip_centered_quat )[1],  # 9
                            self.fingertip_centered_linvel,  # 3
                            self.fingertip_centered_angvel,  # 3
                            
                            self.pose_world_to_robot_base( self.gripper_goal_pos, self.gripper_goal_quat)[0],  # 3
                            self.pose_world_to_robot_base( self.gripper_goal_pos, self.gripper_goal_quat)[1],  # 9
                            self.pose_world_to_robot_base( self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[0],  # 3
                            self.pose_world_to_robot_base( self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat)[1],  # 9
                            delta_pos,  # 3
                            delta_quat,  # 4

                            self.pose_world_to_robot_base( self.plug_pos, self.plug_quat )[0],  # 3
                            self.pose_world_to_robot_base( self.plug_pos, self.plug_quat )[1],  # 9
                            self.pose_world_to_robot_base( self.socket_pos, self.socket_quat )[0],  # 3
                            self.pose_world_to_robot_base( self.socket_pos, self.socket_quat )[1],  # 9
                            delta_plug_pos,  # 3
                            delta_plug_quat,  # 4

                            self.pose_world_to_robot_base( self.noisy_plug_pos, self.noisy_plug_quat )[0],  # 3
                            self.pose_world_to_robot_base( self.noisy_plug_pos, self.noisy_plug_quat )[1],  # 9
                            self.pose_world_to_robot_base( self.noisy_socket_pos, self.noisy_socket_quat )[0],  # 3
                            self.pose_world_to_robot_base( self.noisy_socket_pos, self.noisy_socket_quat )[1],  # 9
                            noisy_delta_plug_pos,  # 3
                            noisy_delta_plug_quat,  # 4

                            noisy_delta_pos - delta_pos, # 3
                            noisy_delta_plug_pos - delta_plug_pos, # 3
                    
                            torch_jit_utils.quat_mul(noisy_delta_quat, torch_jit_utils.quat_conjugate(delta_quat)), # 4
                            torch_jit_utils.quat_mul(noisy_delta_plug_quat, torch_jit_utils.quat_conjugate(delta_plug_quat)), # 4
                                                                                                                                     ] # 139 
        '''
        state_tensors = [
                    self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
                    self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 9
                    self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[0],  # 3
                    self.pose_world_to_robot_base(self.socket_pos, self.socket_quat)[1],  # 9
                                                                                                                                ] # 24



        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)
        return self.obs_buf
    

    def compute_reward(self):
        
        """
            Detect successes and failures. 
            Update reward and reset buffers.
        """

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_rew_buf(self):
        
        """
            Compute reward at current timestep.
        """

        action_penalty = torch.norm(self.actions, p=2, dim=-1)
        action_reward = self.cfg_task.rl.action_penalty_scale * action_penalty

        action_delta_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        action_delta_reward = self.cfg_task.rl.action_delta_scale * action_delta_penalty

        plug_ori_penalty = torch.norm(self.plug_quat - self.identity_quat, p=2, dim=-1)
        ori_reward = plug_ori_penalty * self.cfg_task.rl.ori_reward_scale

        keypoint_dist = self._get_keypoint_dist()
        keypoint_reward = keypoint_dist * self.cfg_task.rl.keypoint_reward_scale

        is_plug_engaged_w_socket = self._check_plug_engaged_w_socket()
        engagement = self._get_engagement_reward_scale(is_plug_engaged_w_socket, self.cfg_task.rl.success_height_thresh)
        engagement_reward = engagement * self.cfg_task.rl.engagement_reward_scale
        
        verbose = True
        if verbose:
            print("reward", keypoint_reward[0], engagement_reward[0])
        
        # self.rew_buf[:] = ori_reward

        self.rew_buf[:] = keypoint_reward + engagement_reward # + ori_reward # + action_reward + action_delta_reward

        distance_reset_buf = (self.far_from_goal_buf | self.degrasp_buf)
        early_reset_reward = distance_reset_buf * self.cfg_task.rl.early_reset_reward_scale
        self.rew_buf[:] += (early_reset_reward)

        # self.rew_buf[:] += (self.timeout_reset_buf * self.success_reset_buf) * self.cfg_task.rl.success_bonus
        self.extras['successes'] = ((self.timeout_reset_buf | distance_reset_buf) * self.success_reset_buf) * 1.0
        self.extras['keypoint_reward'] = keypoint_reward
        self.extras['engagement_reward'] = engagement_reward
        self.extras['ori_reward'] = ori_reward

        self.reward_log_buf[:] = self.rew_buf[:]

        # self.ep_success_count[self.ep_count < 3] += self.success_reset_buf[self.ep_count < 3] * 1
        # self.ep_count[self.ep_count < 3] += self.reset_buf[self.ep_count < 3] * 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            # print(torch.sum(self.ep_count).item(), self.ep_success_count)
            # print('Success Rate:', (torch.sum(self.ep_success_count)/torch.sum(self.ep_count)))
            # self.ep_count[:] = 0
            # self.ep_success_count[:] = 0
            if not self.cfg_task.data_logger.collect_data:
                success_dones = self.success_reset_buf.nonzero()
                failure_dones = (1.0 - self.success_reset_buf).nonzero()

                print(
                        'Success Rate:', torch.mean(self.success_reset_buf * 1.0).item(),
                        'Avg Ep Reward:', torch.mean(self.reward_log_buf).item(),
                        ' Success Reward:', self.rew_buf[success_dones].mean().item(),
                        ' Failure Reward:', self.rew_buf[failure_dones].mean().item()
                                                                                                )

    def _update_reset_buf(self):
        
        """
            Assign environments for reset if successful or failed.
        """

        # if successfully inserted to a certain threshold
        self.success_reset_buf[:] = self._check_plug_inserted_in_socket()

        # if we are collecting data, reset at insertion
        # if self.cfg_task.data_logger.collect_data or self.cfg_task.data_logger.collect_test_sim:
        #     self.reset_buf[:] |= self.success_reset_buf[:]

        # If max episode length has been reached
        self.timeout_reset_buf[:] = torch.where(
                                                    self.progress_buf[:] >= (self.cfg_task.rl.max_episode_length - 1),
                                                    torch.ones_like(self.reset_buf),
                                                    self.reset_buf
                                                                                                                            )

        self.reset_buf[:] = self.timeout_reset_buf[:]

        # check is object is grasped and reset if not
        roll, pitch, _ = get_euler_xyz(self.plug_quat.clone())
        roll[roll > np.pi] -= 2 * np.pi
        pitch[pitch > np.pi] -= 2 * np.pi
        self.degrasp_buf[:] = (torch.abs(roll) > 0.3) | (torch.abs(pitch) > 0.3)

        # check if object is too far from gripper
        fingertips_plug_dist = (
                                    torch.norm(self.left_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12) | (
                                    torch.norm(self.right_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12) | (
                                        torch.norm(self.middle_finger_pos - self.plug_pos, p=2, dim=-1) > 0.12
                                                                                                                        )
        
        # self.degrasp_buf[:] |= fingertips_plug_dist

        # Reset at grasping fails
        # self.reset_buf[:] |= self.degrasp_buf[:]

        # If plug is too far from socket pos
        self.dist_plug_socket = torch.norm(self.plug_pos - self.socket_pos, p=2, dim=-1)
        self.far_from_goal_buf[:] = self.dist_plug_socket > 0.2  # self.cfg_task.rl.far_error_thresh,
        # self.reset_buf[:] |= self.far_from_goal_buf[:]

    def _reset_environment(self, env_ids):

        random_init_idx = {}
        
        for subassembly in self.cfg_env.env.desired_subassemblies:
            random_init_idx[subassembly] = torch.randint(0, self.total_init_poses[subassembly], size=(len(env_ids),))
        subassemblies = [self.envs_asset[e_id] for e_id in range(self.num_envs)]

        kuka_dof_pos = torch.zeros((len(env_ids), 15))
        socket_pos = torch.zeros((len(env_ids), 3))
        socket_quat = torch.zeros((len(env_ids), 4))
        plug_pos = torch.zeros((len(env_ids), 3))
        plug_quat = torch.zeros((len(env_ids), 4))

        # socket around within x[-1cm, 1cm], y[-1cm, 1cm], z[-2mm, 3mm]
        socket_pos[:, 0] = 0.5  
        socket_pos[:, 1] = 0.0
        socket_pos[:, 2] = 0.0
        socket_pos_noise = np.random.uniform(-0.01, 0.01, 3)
        socket_pos_noise[2] = np.random.uniform(-0.002, 0.003, 1)
        socket_pos[:, :] += torch.from_numpy(socket_pos_noise)

        socket_euler_w_noise = np.array([0, 0, 0])
        socket_euler_w_noise[2] = np.random.uniform(-0.035, 0.035, 1) # -2 to 2 degrees
        socket_quat[:, :] = torch.from_numpy(R.from_euler('xyz', socket_euler_w_noise).as_quat())
        
        # above socket with overlap
        plug_pos = socket_pos.clone()
        plug_pos_noise = torch.rand((len(env_ids), 3)) * 0.0254 # 0 to 0.0254
        plug_pos_noise[:, 2] = ((torch.rand((len(env_ids), )) * (0.007 - 0.003)) + 0.003) + 0.02 # 0.003 to 0.01
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(plug_pos_noise[:, 0], plug_pos_noise[:, 1], plug_pos_noise[:, 2], c='r', marker='o')
        # plt.show()
        plug_pos[:, :] += plug_pos_noise
        plug_quat[:, -1] = 1.

        for i, e in enumerate(env_ids):
            subassembly = subassemblies[e]
            kuka_dof_pos[i] = self.init_dof_pos[subassembly][random_init_idx[subassembly][i]]
            # plug_pos_noise = np.random.uniform(0.0, 0.0254, 3)
            # plug_pos[i, :2] += torch.from_numpy(plug_pos_noise[:2]).to(self.device)

            # socket_pos[i] = self.init_socket_pos[subassembly][random_init_idx[subassembly][i]]
            # socket_pos[i, :3] += socket_pos_noise
            # socket_quat[i] = self.init_socket_quat[subassembly][random_init_idx[subassembly][i]]
            # plug_pos[i] = self.init_plug_pos[subassembly][random_init_idx[subassembly][i]]
            # plug_pos[i, 2] -= 0.0025
            # # plug_euler = np.random.uniform(-np.pi, np.pi, 1)
            # # plug_quat[i] = torch.from_numpy(R.from_euler('xyz', [0, 0, plug_euler]).as_quat())
            # plug_quat[i] = self.init_plug_quat[subassembly][random_init_idx[subassembly][i]]

        kuka_dof_pos[:, 7:] = 0.
        self._reset_kuka(env_ids, new_pose=kuka_dof_pos)


        for _, v  in self.all_rendering_camera.items():
            self.init_plug_pos_cam[v[0], :] = plug_pos[v[0], :]

        object_pose = {
                        'socket_pose': socket_pos,
                        'socket_quat': socket_quat,
                        'plug_pose': plug_pos,
                        'plug_quat': plug_quat
                                                        }

        self._reset_object(env_ids, new_pose=object_pose)

        for k, v in self.subassembly_extrinsic_contact.items():
            v.reset_socket_pos(socket_pos=socket_pos[0])

        # self._open_gripper(torch.arange(self.num_envs))
        # self._simulate_and_refresh()


    def reset_idx(self, env_ids):
        
        """
            Reset specified environments.
        """
        
        # self.test_plot = []
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # self.rew_buf[:] = 0
        # self._reset_kuka(env_ids)
        # self._reset_object(env_ids)
        # TODO: change this to reset dof and reset root states
        
        self.disable_gravity()

        self._reset_environment(env_ids)

        # # Move arm to grasp pose
        plug_pos_noise = (2 * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)) * 0.005 # self.cfg_task.randomize.grasp_plug_noise
        first_plug_pose = self.plug_grasp_pos.clone()
        # first_plug_pose[:, 0] -= 0.005
        # first_plug_pose[:, 1] += plug_pos_noise[:, 1]
        first_plug_pose[:, :] += plug_pos_noise[:, :]

        self._move_arm_to_desired_pose(env_ids, first_plug_pose, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps*2)


        # self._zero_velocities(env_ids)
        self._refresh_task_tensors()
        
        self._close_gripper(torch.arange(self.num_envs))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_plug_pos = self.plug_pos.clone().cpu().numpy()
        # ax.scatter(plot_plug_pos[:, 0], plot_plug_pos[:, 1], plot_plug_pos[:, 2], c='r', marker='o')
        # plt.show()
        
        eef_pos = torch.cat(self.pose_world_to_robot_base(self.fingertip_centered_pos.clone(), self.fingertip_centered_quat.clone()), dim=-1)
        self.eef_pos = eef_pos
        self.enable_gravity()

        # print(self.ee
        # self.gym.simulate(self.sim)
        # self.render()
        # self._zero_velocities(env_ids)
        self._zero_velocities(env_ids)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        if self.cfg_task.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                # print('Saving video')
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if self.cfg_task.env.record_ft and 0 in env_ids:
            if self.complete_ft_frames is None:
                self.complete_ft_frames = []
            else:
                self.complete_ft_frames = self.ft_frames[:]
            self.ft_frames = []

        self._reset_buffers(env_ids)


    def _reset_kuka(self, env_ids, new_pose=None):
        
        """
            Reset DOF states and DOF targets of kuka.
        """

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # self.dof_pos[env_ids, :] = new_pose.to(device=self.device)  # .repeat((len(env_ids), 1))

        self.dof_pos[env_ids, :7] = torch.tensor(self.cfg_task.randomize.kuka_arm_initial_dof_pos, device=self.device).repeat((len(env_ids), 1))

        # dont play with these joints (no actuation here)#
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_1_1')] = self.cfg_task.env.openhand.base_angle
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_2_1')] = -self.cfg_task.env.openhand.base_angle
        # dont play with these joints (no actuation here)#

        self.dof_pos[env_ids, list(self.dof_dict.values()).index('finger_1_1_to_finger_1_2')] = self.cfg_task.env.openhand.proximal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('finger_2_1_to_finger_2_2')] = self.cfg_task.env.openhand.proximal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_3_2')] = self.cfg_task.env.openhand.proximal_open + 0.15

        self.dof_pos[env_ids, list(self.dof_dict.values()).index('finger_1_2_to_finger_1_3')] = self.cfg_task.env.openhand.distal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('finger_2_2_to_finger_2_3')] = self.cfg_task.env.openhand.distal_open
        self.dof_pos[env_ids, list(self.dof_dict.values()).index('finger_3_2_to_finger_3_3')] = self.cfg_task.env.openhand.distal_open

        # Stabilize!
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[env_ids] = 0.0  # shape = (num_envs, num_dofs)

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        self.ctrl_target_gripper_dof_pos[env_ids] = self.dof_pos[env_ids, 7:]

        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
                                                self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32)
                                                                                                )
        
        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
                                                            self.sim,
                                                            gymtorch.unwrap_tensor(self.dof_torque),
                                                            gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                            len(multi_env_ids_int32),
                                                                                                                        )

        # Simulate one step to apply changes
        # self._simulate_and_refresh()

    def _reset_object(self, env_ids, new_pose=None):
        
        """
            Reset root state of plug.
        """

        # Randomize root state of plug
        # plug_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        # plug_noise_xy = plug_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.plug_pos_xy_noise, device=self.device))

        # self.root_pos[env_ids, self.plug_actor_id_env, 0] = self.cfg_task.randomize.plug_pos_xy_initial[0] + plug_noise_xy[env_ids, 0]
        # self.root_pos[env_ids, self.plug_actor_id_env, 1] = self.cfg_task.randomize.plug_pos_xy_initial[1] + plug_noise_xy[env_ids, 1]
        # self.root_pos[env_ids, self.plug_actor_id_env, 2] = self.cfg_base.env.table_height
        # self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat(len(env_ids), 1)

        plug_pose = new_pose['plug_pose']
        # plug_pose[:, 0] += 0.1
        plug_quat = new_pose['plug_quat']

        self.root_pos[env_ids, self.plug_actor_id_env, :] = plug_pose.to(device=self.device)
        self.root_quat[env_ids, self.plug_actor_id_env, :] = plug_quat.to(device=self.device)

        # Stabilize plug
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        # self.gym.set_actor_root_state_tensor_indexed(
        #                                                   self.sim,
        #                                                   gymtorch.unwrap_tensor(self.root_state),
        #                                                   gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
        #                                                   len(plug_actor_ids_sim_int32[env_ids])
        #                                                                                                                   )

        # self._simulate_and_refresh()

        # Randomize root state of socket
        # socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        # socket_noise_xy = socket_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, device=self.device))

        # self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.cfg_task.randomize.socket_pos_xy_initial[0] + socket_noise_xy[env_ids, 0]
        # self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.cfg_task.randomize.socket_pos_xy_initial[1] + socket_noise_xy[env_ids, 1]
        # self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height

        # self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat(len(env_ids), 1)

        socket_pose = new_pose['socket_pose']
        socket_quat = new_pose['socket_quat']

        self.root_pos[env_ids, self.socket_actor_id_env, :] = socket_pose.to(device=self.device)
        self.root_quat[env_ids, self.socket_actor_id_env, :] = socket_quat.to(device=self.device)

        # Stabilize socket
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim_int32 = self.socket_actor_ids_sim.to(dtype=torch.int32, device=self.device)

        # print(torch.cat([plug_actor_ids_sim_int32[env_ids], socket_actor_ids_sim_int32[env_ids]]), plug_actor_ids_sim_int32, socket_actor_ids_sim_int32)
        # print(self.root_state[:, plug_actor_ids_sim_int32, :])
        # print(self.root_state[:, socket_actor_ids_sim_int32, :])

        self.gym.set_actor_root_state_tensor_indexed(
                                                        self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state),
                                                        gymtorch.unwrap_tensor(torch.cat([plug_actor_ids_sim_int32[env_ids],socket_actor_ids_sim_int32[env_ids]])),
                                                        len(torch.cat([plug_actor_ids_sim_int32[env_ids],socket_actor_ids_sim_int32[env_ids]]))
                                                                                                                                                                            )

        # Simulate one step to apply changes
        self._simulate_and_refresh()

    def _move_arm_to_desired_pose(self, env_ids, desired_pos, desired_rot=None, sim_steps=30):
        
        """
            Move gripper to desired pose.
        """

        # Set target pos above object
        self.ctrl_target_fingertip_centered_pos[env_ids] = desired_pos[env_ids].clone()

        # Set target rot
        if desired_rot is None:
            ctrl_target_fingertip_centered_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial, device=self.device).unsqueeze(0).repeat(len(env_ids), 1)

            self.ctrl_target_fingertip_centered_quat[env_ids] = torch_jit_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_centered_euler[:, 0],
                ctrl_target_fingertip_centered_euler[:, 1],
                ctrl_target_fingertip_centered_euler[:, 2])
        else:
            self.ctrl_target_fingertip_centered_quat[env_ids] = desired_rot[env_ids]

        # Step sim and render
        for _ in range(sim_steps):
            self._simulate_and_refresh()

            # NOTE: midpoint is calculated based on the midpoint between the actual gripper finger pos,
            # and centered is calculated with the assumption that the gripper fingers are perfectly closed at center.
            # since the fingertips are underactuated, thus we cant know the true pose

            pos_error, axis_angle_error = fc.get_pose_error(
                                                                fingertip_midpoint_pos=self.fingertip_centered_pos,
                                                                fingertip_midpoint_quat=self.fingertip_centered_quat,
                                                                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
                                                                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
                                                                jacobian_type=self.cfg_ctrl['jacobian_type'],
                                                                rot_error_type='axis_angle'
                                                                                                                                                        )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            # Apply the action, keep fingers in the same status
            self.ctrl_target_dof_pos[:, 7:] = self.ctrl_target_gripper_dof_pos
            self._apply_actions_as_ctrl_targets(actions=actions, ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos, do_scale=False)

        # Stabilize
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids, :])
        self.dof_torque[env_ids, :] = torch.zeros_like(self.dof_torque[env_ids, :])

        # Set DOF state
        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
                                                    self.sim,
                                                    gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                    len(multi_env_ids_int32)
                                                                                                        )

        self._simulate_and_refresh()

    def _simulate_and_refresh(self):
        
        """
            Simulate one step, refresh tensors, and render results.
        """
        
        self.gym.simulate(self.sim)
        self.render()
        self._refresh_task_tensors()

    def _reset_buffers(self, env_ids):
        
        """
            Reset buffers. 
        """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.time_complete_task = torch.zeros_like(self.progress_buf)
        self.rew_buf[env_ids] = 0
        self.reward_log_buf[env_ids] = 0

        self.degrasp_buf[env_ids] = 0
        self.success_reset_buf[env_ids] = 0
        self.far_from_goal_buf[env_ids] = 0
        self.timeout_reset_buf[env_ids] = 0

        self.plug_socket_dist[env_ids, ...] = 0.

        if self.cfg_task.env.compute_contact_gt:
            self.gt_extrinsic_contact[env_ids] *= 0

        if 0 in env_ids:
            for keys in self.for_plots.keys():
                self.for_plots[keys] = []

    def _set_viewer_params(self):
        
        """
            Set viewer parameters.
        """
        
        bx, by, bz = 0.5, 0.0, 0.05
        cam_pos = gymapi.Vec3(bx - 0.1, by - 0.1, bz + 0.07)
        cam_target = gymapi.Vec3(bx, by, bz)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        
        """
            Apply actions from policy as position/rotation targets.
        """

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                                                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                rot_actions_quat,
                                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,1))
        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)
        # TODO should be changed to delta as well ?
        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos  # directly putting desired gripper_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, env_ids, sim_steps=20):
        
        """
            Fully open gripper using controller. 
            Called outside RL loop (i.e., after last step of episode).
        """

        gripper_dof_pos = 0 * self.gripper_dof_pos.clone()

        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_1_1_to_finger_1_2') - 7] = self.cfg_task.env.openhand.proximal_open
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_2_1_to_finger_2_2') - 7] = self.cfg_task.env.openhand.proximal_open
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_3_2') - 7] = self.cfg_task.env.openhand.proximal_open

        self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=gripper_dof_pos, sim_steps=sim_steps)
        self.ctrl_target_gripper_dof_pos = gripper_dof_pos

    def _close_gripper(self, env_ids, sim_steps=20):
        
        """
            Fully close gripper using controller. 
            Called outside RL loop (i.e., after last step of episode).
        """

        gripper_dof_pos = self.gripper_dof_pos.clone()
        gripper_dof_pos[env_ids,list(self.dof_dict.values()).index('base_to_finger_1_1') - 7] = self.cfg_task.env.openhand.base_angle
        gripper_dof_pos[env_ids,list(self.dof_dict.values()).index('base_to_finger_2_1') - 7] = -self.cfg_task.env.openhand.base_angle

        gripper_proximal_close_noise = np.random.uniform(0.0, 0.01, 3)
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_1_1_to_finger_1_2') - 7] = self.cfg_task.env.openhand.proximal_close + gripper_proximal_close_noise[0]
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_2_1_to_finger_2_2') - 7] = self.cfg_task.env.openhand.proximal_close + gripper_proximal_close_noise[1]
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('base_to_finger_3_2') - 7]       = self.cfg_task.env.openhand.proximal_close + gripper_proximal_close_noise[2]

        gripper_distal_close_noise = np.random.uniform(0., 0.005, 3)
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_1_2_to_finger_1_3') - 7] = self.cfg_task.env.openhand.distal_close + gripper_distal_close_noise[0]
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_2_2_to_finger_2_3') - 7] = self.cfg_task.env.openhand.distal_close + gripper_distal_close_noise[1]
        gripper_dof_pos[env_ids, list(self.dof_dict.values()).index('finger_3_2_to_finger_3_3') - 7] = self.cfg_task.env.openhand.distal_close + gripper_distal_close_noise[2]
        
        # slowly grasp the plug
        for i in range(100):
            diff = gripper_dof_pos[env_ids, :] - self.gripper_dof_pos[env_ids, :]
            self.ctrl_target_gripper_dof_pos = self.gripper_dof_pos[env_ids, :] + diff * 0.1
            # print(self.ctrl_target_gripper_dof_pos)
            self._move_gripper_to_dof_pos(env_ids=env_ids, gripper_dof_pos=self.ctrl_target_gripper_dof_pos, sim_steps=1)

        # allows for inward squeeze to maintain stable grasp
        # self.ctrl_target_gripper_dof_pos = gripper_dof_pos
        # print(self.ctrl_target_gripper_dof_pos)
            
    def _move_gripper_to_dof_pos(self, env_ids, gripper_dof_pos, sim_steps=20):
        
        """
            Move gripper fingers to specified DOF position using controller.
        """

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)  # no arm motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self._simulate_and_refresh()

    def _lift_gripper(self, env_ids, gripper_dof_pos, lift_distance=0.2, sim_steps=20):
        
        """
            Lift gripper by specified distance. 
            Called outside RL loop (i.e., after last step of episode).
        """

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[env_ids, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self._simulate_and_refresh()

    def _get_keypoint_offsets(self, num_keypoints):
        
        """
            Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        """

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device)  # - 0.5
        return keypoint_offsets

    def _get_keypoint_dist(self):
        
        """
            Get keypoint distances.
        """

        keypoint_dist = torch.sum(torch.norm(self.keypoints_socket - self.keypoints_plug, p=2, dim=-1), dim=-1)
        return keypoint_dist

    def _zero_velocities(self, env_ids):

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        # Set DOF state
        multi_env_ids_int32 = self.kuka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
                                                self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32)
                                                                                                    )

        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
                                                        self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state),
                                                        gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
                                                        len(plug_actor_ids_sim_int32[env_ids])
                                                                                                                                    )

        # self._simulate_and_refresh()

    def _check_lift_success(self, height_multiple):
        
        """
            Check if plug is above table by more than specified multiple times height of plug.
        """

        # print('plug height', self.plug_pos[:, 2], self.cfg_base.env.table_height + self.plug_heights.squeeze(-1) * height_multiple)
        lift_success = torch.where(
            self.plug_pos[:, 2] > self.cfg_base.env.table_height + self.plug_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success

    def _check_plug_close_to_socket(self):
        
        """
            Check if plug is close to socket.
        """
        
        return torch.norm(self.plug_pos[:, :2] - self.socket_tip[:, :2], p=2, dim=-1) < self.cfg_task.rl.close_error_thresh

    def _check_plug_inserted_in_socket(self):
        
        """
            Check if plug is inserted in socket.
        """

        # Check if plug is within threshold distance of assembled state
        is_plug_below_insertion_height = (self.plug_pos[:, 2] <= (self.socket_tip[:, 2] - self.cfg_task.rl.success_height_thresh))
        # Check if plug is close to socket
        # NOTE: This check addresses edge case where plug is within threshold distance of
        # assembled state, but plug is outside socket
        is_plug_close_to_socket = self._check_plug_close_to_socket()
        # Combine both checks
        is_plug_inserted_in_socket = torch.logical_and(is_plug_below_insertion_height, is_plug_close_to_socket)
        return is_plug_inserted_in_socket

    def _check_plug_engaged_w_socket(self):
        
        """
            Check if plug is engaged with socket.
        """

        # Check if base of plug is below top of socket
        # NOTE: In assembled state, plug origin is coincident with socket origin;
        # thus plug pos must be offset to compute actual pos of base of plug
        is_plug_below_engagement_height = ((self.plug_pos[:, 2]) < self.socket_tip[:, 2])

        # Check if plug is close to socket
        # NOTE: This check addresses edge case where base of plug is below top of socket,
        # but plug is outside socket
        is_plug_close_to_socket = self._check_plug_close_to_socket()  # torch.norm(self.plug_pos[:, :2] - self.socket_tip[:, :2], p=2, dim=-1) < 0.005 # self._check_plug_close_to_socket()
        # print(is_plug_below_engagement_height[0], is_plug_close_to_socket[0])

        # Combine both checks
        is_plug_engaged_w_socket = torch.logical_and(is_plug_below_engagement_height, is_plug_close_to_socket)

        return is_plug_engaged_w_socket


    def _get_engagement_reward_scale(self, is_plug_engaged_w_socket, success_height_thresh):
        
        """
            Compute scale on reward. 
            If plug is not engaged with socket, 
                                                scale is zero.
            If plug is engaged, 
                                scale is proportional to distance between plug and bottom of socket.
        """

        # Set default value of scale to zero
        reward_scale = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        
        # For envs in which plug and socket are engaged, compute positive scale
        engaged_idx = np.argwhere(is_plug_engaged_w_socket.cpu().numpy().copy()).squeeze()
        height_dist = self.plug_pos[engaged_idx, 2] - self.socket_pos[engaged_idx, 2]
        
        # NOTE: Edge case: if success_height_thresh is greater than 0.1,
        # denominator could be negative
        reward_scale[engaged_idx] = 1.0 / ((height_dist - success_height_thresh) + 0.1)
        
        return reward_scale