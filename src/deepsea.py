import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from bsuite.environments.base import Environment
from dm_env import specs

import experiment
from aapi import AAPI, Action, BasisFunction, Features, Observation

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_integer("size", 20, "Size of deep sea environment.")
flags.DEFINE_integer("train_steps", 20000, "Number of train steps.")
flags.DEFINE_integer("train_every", 50, "Period of train steps.")
flags.DEFINE_float("learning_rate", .1, "Learning rate.")
flags.DEFINE_integer("buffer_size", 500, "Set to -1 if arbitrary size.")
flags.DEFINE_integer("weight_size", 30, "Number of weights to select.")
flags.DEFINE_string("weight_select", 'last', "Weight selection method.")


class DeepSea(Environment):
    def __init__(self, size: int):
        super().__init__()
        self._size = size
        self._t = 0
        self._total_cost = 0
        self._column = 0
        self._row = 0
        self._reset()

    def _get_observation(self):
        obs = np.zeros(shape=(2 * self._size, ), dtype=np.float32)
        obs[self._row] = 1.
        obs[self._size + self._column] = 1.
        return obs

    def _reset(self) -> dm_env.TimeStep:
        self._row = 0
        self._column = 0
        self._timestep = 0
        return dm_env.restart(self._get_observation())

    def _step(self, action: int) -> dm_env.TimeStep:

        self._row = (self._row + 1) % self._size
        self._column = (self._column + 2 * action - 1) % self._size

        reward = -float(action)
        if (self._row == self._size - 1) and (self._column == self._size - 1):
            reward = 2 * self._size

        self._t += 1
        self._total_cost -= reward

        observation = self._get_observation()

        return dm_env.transition(reward=reward, observation=observation)

    def observation_spec(self):
        return specs.Array(shape=(2 * self._size, ), dtype=np.float32)

    def action_spec(self):
        return specs.DiscreteArray(2, name='action')

    @property
    def _avg_cost(self):
        return round(self._total_cost / self._t, 2)

    def bsuite_info(self):
        return dict(avg_cost=self._avg_cost)


def get_basis(size: int) -> BasisFunction:

    nS, nA = size, 2

    # basis function
    def basis_function(
        o: Observation,
        a: Action,
    ) -> Features:
        f = jnp.zeros((nA, 2 * nS))
        f = jax.ops.index_update(f, jax.ops.index[a], o)
        return f.flatten()

    return basis_function


def main(unused_arg):

    key = jax.random.PRNGKey(FLAGS.seed)
    rng = hk.PRNGSequence(key)

    # set environment
    env = DeepSea(size=FLAGS.size)

    # set basis function
    basis_function = get_basis(FLAGS.size)

    # set agent
    agent = AAPI(obs_spec=env.observation_spec(),
                 action_spec=env.action_spec(),
                 basis_function=basis_function,
                 train_every=FLAGS.train_every,
                 learning_rate=FLAGS.learning_rate,
                 buffer_size=FLAGS.buffer_size,
                 weight_size=FLAGS.weight_size,
                 weight_select=FLAGS.weight_select,
                 rng=rng)

    # run experiment
    experiment.run(agent=agent,
                   environment=env,
                   num_steps=FLAGS.train_steps,
                   verbose=True)


if __name__ == "__main__":
    app.run(main)
