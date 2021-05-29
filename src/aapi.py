from collections import deque, namedtuple
from functools import partial
from typing import Callable, Optional, Tuple, Union

import dm_env
import haiku as hk
import jax
import numpy as np
from bsuite.baselines.base import Agent
from dm_env import specs
from jax import numpy as jnp

Observation = np.ndarray
Action = int
Features = Union[np.ndarray, jnp.ndarray]
BasisFunction = Callable[[Observation, Action], Features]
AgentState = namedtuple('AgentState', "seq_weights")


@jax.jit
def _eval(
    agent_state: AgentState,
    batch_f: jnp.ndarray,
    batch_r: jnp.ndarray,
) -> AgentState:
    batch_rev_q = jnp.cumsum(batch_r[::-1]) / jnp.arange(1, 1 + len(batch_r))
    batch_q = batch_rev_q[::-1]
    weights = jnp.linalg.lstsq(batch_f, batch_q)[0]
    return agent_state._replace(seq_weights=agent_state.seq_weights +
                                [weights])


@partial(jax.jit, static_argnums=(4, ))
def _select_last(
    agent_state: jnp.ndarray,
    batch_f: jnp.ndarray,
    lr: float,
    key: np.ndarray,
    size: Optional[int] = None,
) -> Action:
    w_all = agent_state.seq_weights
    if size is not None:
        w_all = w_all[-(size + 1):]
    if len(w_all) == 1:
        q = jnp.matmul(batch_f, w_all[0])
        div = jnp.linalg.norm(q - jnp.mean(q), ord='inf')
        logits = 2 * q
        temperature = lr * jnp.sqrt(2) * div
    else:
        q_all = jnp.matmul(jnp.stack(w_all), batch_f.T)
        div = jnp.linalg.norm(q_all[1:] - q_all[:-1], ord='inf', axis=1)
        logits = q_all[1:].sum(axis=0) + q_all[-1]
        temperature = lr * jnp.sqrt(2 * jnp.power(div, 2).sum())
    action = jax.random.categorical(key, logits / temperature)
    return action


@partial(jax.jit, static_argnums=(4, ))
def _select_random(
    agent_state: jnp.ndarray,
    batch_f: jnp.ndarray,
    lr: float,
    key: np.ndarray,
    size: Optional[int] = None,
) -> Action:
    w_all = agent_state.seq_weights
    q = jnp.matmul(batch_f, w_all[-1])
    if (size is not None) and len(w_all) > size:
        idxs = jax.random.choice(key,
                                 jnp.arange(1, len(w_all)), (size, ),
                                 replace=False)
        scale = jnp.sqrt(len(w_all) / size)
    else:
        idxs = jnp.arange(1, len(w_all))
        scale = 1
    if len(w_all) == 1:
        div = jnp.linalg.norm(q - jnp.mean(q), ord='inf')
        logits = 2 * q
        temperature = lr * jnp.sqrt(2) * div
    else:
        w = jnp.stack(w_all)[idxs]
        w_ = jnp.stack(w_all)[idxs - 1]
        q_all = jnp.matmul(jnp.stack(w), batch_f.T)
        q_all_ = jnp.matmul(jnp.stack(w_), batch_f.T)
        div = jnp.linalg.norm(q_all - q_all_, ord='inf', axis=1)
        logits = q_all.sum(axis=0) + q
        temperature = lr * jnp.sqrt(2 * jnp.power(div, 2).sum())
        temperature *= scale
    action = jax.random.categorical(key, logits / temperature)
    return action


class Buffer:
    def __init__(self, capacity: Optional[int] = None):

        self._observations = deque(maxlen=capacity)
        self._actions = deque(maxlen=capacity)
        self._rewards = deque(maxlen=capacity)

    def append(self, timestep: dm_env.TimeStep, action: Action,
               new_timestep: dm_env.TimeStep):

        self._observations.append(timestep.observation)
        self._actions.append(action)
        self._rewards.append(new_timestep.reward)

        if new_timestep.last():
            raise AssertionError(
                'Environment should have infinite horizon (see paper).')

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_o = np.stack(self._observations)
        batch_a = np.stack(self._actions)
        batch_r = np.stack(self._rewards)
        return batch_o, batch_a, batch_r


class AAPI(Agent):
    def __init__(self,
                 obs_spec: specs.Array,
                 action_spec: specs.DiscreteArray,
                 basis_function: BasisFunction,
                 train_every: int,
                 learning_rate: float,
                 buffer_size: int,
                 rng: hk.PRNGSequence,
                 weight_select: str = 'last',
                 weight_size: Optional[int] = None):
        assert weight_select in [
            'random', 'last'
        ], "'weight_select' take values in ['random', 'last'] "
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._basis_function = jax.vmap(basis_function)
        self._train_every = train_every
        self._learning_rate = learning_rate
        self._rng = rng
        self._weight_size = weight_size

        if weight_select == 'random':
            self._select = _select_random
        else:
            self._select = _select_last

        # initialize buffer
        capacity = buffer_size if buffer_size > 0 else None
        self._t = 0
        self._buffer = Buffer(capacity)

        # initialize parameters
        self._state = AgentState(seq_weights=[])

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: Action,
        new_timestep: dm_env.TimeStep,
    ):
        self._t += 1
        self._buffer.append(timestep, action, new_timestep)
        if self._t % self._train_every == 0:
            batch_o, batch_a, batch_r = self._buffer.get()
            batch_f = self._basis_function(batch_o, batch_a)
            self._state = _eval(self._state, batch_f, batch_r)

    def select_action(
        self,
        timestep: dm_env.TimeStep,
    ) -> Action:
        nA = self._action_spec.num_values
        if len(self._state.seq_weights) == 0:
            action = jax.random.randint(next(self._rng), (), 0, nA)
            return int(action)
        lr = self._learning_rate
        o = timestep.observation
        batch_o = jnp.repeat(np.expand_dims(o, 0), nA, axis=0)
        batch_a = jnp.arange(nA)
        batch_f = self._basis_function(batch_o, batch_a)
        action = self._select(self._state, batch_f, lr, next(self._rng),
                              self._weight_size)
        return int(action)
