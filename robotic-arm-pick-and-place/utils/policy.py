"""Policy lifecycle and action-chunk isolation without privileged observations."""

from collections import deque
import inspect
import numpy as np


class PolicySession:
    """Wrap a callable or predict(obs) policy; clear chunks at every episode.

    A stateful policy should provide reset(seed=...), reset(), or be constructed
    by a fresh per-episode factory. Reset never receives task IDs or simulator
    info. A policy returns one action or a chunk shaped (K, action_dimension).
    Optional observe_action(action) receives each issued command, including
    cached chunk steps, before downstream faults or actuator modifications.
    Optional observe_proprio(proprio) receives each public measured state before
    that step's prediction or cached action. It never receives simulator info.
    """

    def __init__(self, policy, action_space, *, execute_actions=None):
        if execute_actions is not None and (isinstance(execute_actions, bool) or not isinstance(execute_actions, int) or execute_actions < 1):
            raise ValueError("execute_actions must be a positive integer or None")
        self.policy = policy
        self.action_space = action_space
        self.execute_actions = execute_actions
        self.pending = deque()

    def reset(self, seed=None):
        self.pending.clear()
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            signature = inspect.signature(reset)
            accepts_seed = "seed" in signature.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            reset(seed=seed) if accepts_seed else reset()

    def __call__(self, observation):
        observe_proprio = getattr(self.policy, "observe_proprio", None)
        if observe_proprio is not None:
            observe_proprio(np.asarray(observation["proprio"]).copy())
        if not self.pending:
            predict = getattr(self.policy, "predict", self.policy)
            result = np.asarray(predict(observation), dtype=np.float32)
            if result.shape == self.action_space.shape:
                result = result[None]
            if result.ndim != 2 or result.shape[1:] != self.action_space.shape or not len(result):
                raise ValueError("Policy must return an action or a nonempty (K, action_dim) chunk")
            for action in result:
                if not self.action_space.contains(action):
                    raise ValueError("Policy action is nonfinite or outside the declared normalized space")
            for action in result[:self.execute_actions]:
                self.pending.append(action.copy())
        action = self.pending.popleft()
        observe_action = getattr(self.policy, "observe_action", None)
        if observe_action is not None:
            # Command history belongs to the policy. Unused future predictions
            # are not commands, and callback mutation must not change this one.
            observe_action(action.copy())
        return action
