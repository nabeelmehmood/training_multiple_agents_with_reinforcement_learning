from gym.envs.registration import register

register(
    id='MultiAgentEnv-v0',
    entry_point='MultiAgentEnv.envs:MultiClawEnv',
    timestep_limit=2000,
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='ReachEnv-v0',
    entry_point='MultiAgentEnv.envs:ReachEnv',
    timestep_limit=300,
    reward_threshold=1.0,
    nondeterministic = True,
)
