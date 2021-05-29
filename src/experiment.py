# adapted from https://github.com/deepmind/bsuite/blob/master/bsuite/baselines/experiment.py

import dm_env
from bsuite.baselines.base import Agent


def run(agent: Agent,
        environment: dm_env.Environment,
        num_steps: int,
        verbose: bool = False) -> None:

    timestep = environment.reset()

    for t in range(num_steps):
        # Generate an action from the agent's policy.
        action = agent.select_action(timestep)

        # Step the environment.
        new_timestep = environment.step(action)

        # Tell the agent about what just happened.
        agent.update(timestep, action, new_timestep)

        # Book-keeping.
        timestep = new_timestep

        if verbose and (t % agent._train_every == 0):
            bsuite_info = environment.bsuite_info()
            logs = ['step = {}'.format(t)] + [
                '{} = {}'.format(key, item)
                for key, item in bsuite_info.items()
            ]
            print(' | '.join(logs))
