from isaacgyminsertion.pbt.launcher.run_description import ParamGrid, RunDescription, Experiment
from isaacgyminsertion.pbt.experiments.allegro_kuka_pbt_base import kuka_env, kuka_base_cli
from isaacgyminsertion.pbt.experiments.run_utils import version


_pbt_num_policies = 8
_name = f'{kuka_env}_throw_{version}_pbt_{_pbt_num_policies}p'

_params = ParamGrid([
    ('pbt.policy_idx', list(range(_pbt_num_policies))),
])

_wandb_activate = True
_wandb_group = f'pbt_{_name}'
_wandb_entity = 'your_wandb_entity'
_wandb_project = 'your_wandb_project'

cli = kuka_base_cli + \
    f' task=AllegroKukaLSTM ' \
    f'task/env=throw wandb_activate=True pbt.num_policies={_pbt_num_policies} ' \
    f'wandb_project={_wandb_project} wandb_entity={_wandb_entity} wandb_activate={_wandb_activate} wandb_group={_wandb_group}'

RUN_DESCRIPTION = RunDescription(
    f'{_name}',
    experiments=[Experiment(f'{_name}', cli, _params.generate_params(randomize=False))],
    experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
    param_prefix='', customize_experiment_name=False,
)
