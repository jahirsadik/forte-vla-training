"""Public command-history codec for absolute joint-reference learning targets.

The simulator interface remains normalized joint-reference velocity plus nominal
jaw width. This adapter learns normalized absolute arm references, then emits
bounded commands through that existing interface. Its state is reconstructed from
the first public proprioception and policy-issued commands, never simulator state.
Under an unobserved actuator/fault override it tracks *intended* references only.
"""

from copy import deepcopy

import numpy as np

from utils.actions import decode_action


def _limits(contract):
    lower = np.asarray(contract.get("joint_lower_limits_rad"), dtype=np.float64)
    upper = np.asarray(contract.get("joint_upper_limits_rad"), dtype=np.float64)
    if (contract.get("id") != "forte-joint-reference-v2"
            or contract.get("arm_joint_count") != 7 or contract.get("shape") != [8]
            or lower.shape != (7,) or upper.shape != (7,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper)):
        raise ValueError("Absolute-reference targets require explicit finite seven-joint operating bounds")
    for name in ("policy_hz", "joint_reference_speed_rad_s", "gripper_speed_limit_m_s"):
        value = contract.get(name)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid reference action contract: {name}")
    closed, opened = contract.get("gripper_closed_m"), contract.get("gripper_open_m")
    if closed is None or opened is None or not np.isfinite([closed, opened]).all() or closed >= opened:
        raise ValueError("Invalid nominal jaw-width bounds")
    return lower, upper


def normalize_joint_references(references, contract):
    """Map absolute radians to [-1, 1] using public operating bounds only."""
    lower, upper = _limits(contract)
    references = np.asarray(references, dtype=np.float64)
    if (references.ndim < 1 or references.shape[-1] != 7
            or not np.isfinite(references).all()
            or np.any(references < lower) or np.any(references > upper)):
        raise ValueError("Joint-reference targets must be finite and within operating bounds")
    return np.clip(2 * (references - lower) / (upper - lower) - 1, -1, 1).astype(np.float32)


class ReferenceActionCodec:
    """Track issued commands; predicting a chunk never consumes its actions."""

    def __init__(self, contract):
        self.contract = deepcopy(contract)
        self.lower, self.upper = _limits(self.contract)
        self.step_limit = (self.contract["joint_reference_speed_rad_s"]
                           / self.contract["policy_hz"])
        self.reset()

    def reset(self):
        self.arm_reference = None
        self.jaw_reference = None

    def initialize(self, proprio):
        values = np.asarray(proprio, dtype=np.float64)
        if values.shape != (23,) or not np.isfinite(values).all():
            raise ValueError("Reference initialization requires finite public 23D proprioception")
        if self.arm_reference is None:
            # Initial measured positions may differ from ideal reset positions by
            # float32 rounding; use public measurements projected to legal bounds.
            self.arm_reference = np.clip(values[:7], self.lower, self.upper).copy()
            self.jaw_reference = float(np.clip(values[14], self.contract["gripper_closed_m"],
                                               self.contract["gripper_open_m"]))

    def observe_action(self, action):
        """Advance exactly once for a policy-issued command, before hidden faults."""
        if self.arm_reference is None:
            raise ValueError("Initialize reference tracking before observing an issued action")
        self.arm_reference, self.jaw_reference = decode_action(
            action, self.arm_reference, self.jaw_reference, self.contract)

    def decode_chunk(self, targets, proprio):
        """Convert normalized absolute targets to reachable velocity commands.

        Each predicted target is projected from the locally simulated preceding
        command. Unexecuted tails do not modify the persistent reference tracker.
        The jaw coordinate retains the existing nominal-width interpretation.
        """
        targets = np.asarray(targets, dtype=np.float64)
        if (targets.ndim != 2 or targets.shape[1] != 8 or not len(targets)
                or not np.isfinite(targets).all() or np.any(np.abs(targets) > 1)):
            raise ValueError("Expected a finite nonempty normalized (K, 8) target chunk")
        self.initialize(proprio)
        arm, jaw = self.arm_reference.copy(), self.jaw_reference
        output = np.empty(targets.shape, dtype=np.float32)
        for index, target in enumerate(targets):
            desired = self.lower + (target[:7] + 1) * .5 * (self.upper - self.lower)
            command = np.r_[np.clip((desired - arm) / self.step_limit, -1, 1), target[7]].astype(np.float32)
            output[index] = command
            # Project using the emitted float32 values, exactly as the interface
            # will integrate them; do not integrate higher-precision predictions.
            arm, jaw = decode_action(command, arm, jaw, self.contract)
        return output


def reference_targets(initial_proprio, issued_actions, contract):
    """Return post-action arm references plus original nominal jaw commands.

    This pure causal conversion takes no observations after reset, no expert stage,
    no simulator reference fields, and no learned normalization statistics.
    """
    actions = np.asarray(issued_actions, dtype=np.float32)
    if (actions.ndim != 2 or actions.shape[1] != 8
            or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
        raise ValueError("Issued actions must have finite normalized shape (T, 8)")
    codec = ReferenceActionCodec(contract)
    codec.initialize(initial_proprio)
    targets = np.empty_like(actions)
    for index, action in enumerate(actions):
        codec.observe_action(action)
        targets[index, :7] = normalize_joint_references(codec.arm_reference, contract)
        targets[index, 7] = action[7]
    return targets
