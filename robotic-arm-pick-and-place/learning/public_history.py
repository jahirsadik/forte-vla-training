"""Causal public proprioception/issued-command history, shared by data and policy.

Each row precedes the action at that time. References integrate issued commands
from the first measured pose; they are estimates, not simulator controller state.
Hidden actuator overrides are deliberately not incorporated into that estimate.
"""

from collections import deque
import numpy as np

from learning.reference_actions import ReferenceActionCodec, normalize_joint_references

HISTORY_DIM = 49
HISTORY_VERSION = "forte-public-command-history-v1"


def history_spec(steps):
    return dict(version=HISTORY_VERSION, steps=steps, row_dimension=HISTORY_DIM,
                observation_rate="every control step, including cached actions", rgb="current frame only",
                fields=["proprio23", "previous_issued_action8", "estimated_reference8",
                        "estimated_tracking_error8", "observation_valid", "previous_action_valid"],
                proprio="same training-only normalization as the current observation",
                estimated_reference="arm normalized by operating limits; jaw normalized by nominal width bounds",
                estimated_tracking_error="arm radians; jaw error divided by nominal width range",
                padding="left zero padding with validity flags; never repeated future observations",
                reference_initialization="first public measured arm/jaw pose projected to operating bounds",
                command_feedback="issued commands before hidden faults, never unused predicted chunk tails",
                limits="Intended-reference estimate can differ from actual references after hidden overrides")


def normalize_history(history, mean, std):
    values = np.asarray(history, dtype=np.float32).copy()
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if (values.ndim < 2 or values.shape[-1] != HISTORY_DIM or mean.shape != (23,)
            or std.shape != (23,) or not np.isfinite(values).all()
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Invalid public-history normalization inputs")
    values[..., :23] = (values[..., :23] - mean) / std
    values *= values[..., 47:48]  # Invalid left padding stays exactly zero.
    return values


class PublicHistory:
    def __init__(self, contract, steps):
        if type(steps) is not int or not 1 <= steps <= 32:
            raise ValueError("Public history requires 1..32 control steps")
        self.steps = steps
        self.codec = ReferenceActionCodec(contract)
        self.closed = contract["gripper_closed_m"]
        self.width_range = contract["gripper_open_m"] - self.closed
        self.reset()

    def reset(self):
        self.codec.reset()
        self.rows = deque(maxlen=self.steps)
        self.last_action = None
        self.last_proprio = None
        self.awaiting_action = False

    def observe_proprio(self, proprio):
        if self.awaiting_action:
            raise RuntimeError("Missing issued-action feedback before next observation")
        proprio = np.asarray(proprio, dtype=np.float32)
        if proprio.shape != (23,) or not np.isfinite(proprio).all():
            raise ValueError("History accepts finite public 23D proprioception only")
        self.codec.initialize(proprio)
        row = np.zeros(HISTORY_DIM, dtype=np.float32)
        row[:23] = proprio
        if self.last_action is not None:
            row[23:31] = self.last_action
            row[48] = 1
        row[31:38] = normalize_joint_references(self.codec.arm_reference, self.codec.contract)
        row[38] = 2 * (self.codec.jaw_reference - self.closed) / self.width_range - 1
        row[39:46] = self.codec.arm_reference - proprio[:7]
        row[46] = (self.codec.jaw_reference - proprio[14]) / self.width_range
        row[47] = 1
        self.rows.append(row)
        self.last_proprio = proprio.copy()
        self.awaiting_action = True

    def observe_action(self, action):
        if not self.awaiting_action:
            raise RuntimeError("Issued-action feedback requires one preceding observation")
        action = np.asarray(action, dtype=np.float32)
        self.codec.observe_action(action)  # Shared finite/range/shape checks.
        self.last_action = action.copy()
        self.awaiting_action = False

    def raw(self):
        if not self.rows:
            raise RuntimeError("Observe public proprioception before requesting history")
        result = np.zeros((self.steps, HISTORY_DIM), dtype=np.float32)
        result[-len(self.rows):] = np.stack(self.rows)
        return result


def reconstruct_history(proprio, proposed_actions, contract, steps):
    """Pre-action features at t depend only on obs[:t+1] and commands[:t]."""
    proprio = np.asarray(proprio, dtype=np.float32)
    actions = np.asarray(proposed_actions, dtype=np.float32)
    if (actions.ndim != 2 or actions.shape[1] != 8
            or proprio.shape != (len(actions) + 1, 23)
            or not np.isfinite(proprio).all() or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1)):
        raise ValueError("History reconstruction requires T+1 public observations and T issued actions")
    tracker = PublicHistory(contract, steps)
    output = np.empty((len(actions), steps, HISTORY_DIM), dtype=np.float32)
    for t, action in enumerate(actions):
        tracker.observe_proprio(proprio[t])
        output[t] = tracker.raw()
        tracker.observe_action(action)
    return output
