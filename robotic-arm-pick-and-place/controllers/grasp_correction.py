"""Privileged continuation from a sustained target grasp, including before lift.

This is a proactive expert takeover gate, not an autonomous failure detector.
It shares screened physical escape/transport/release with the carry expert.
No live state, object pose, scoring history or actuator limits are changed.
"""

from collections import deque

from controllers.carry_correction import CarryRecoveryExpert


class GraspToGoalExpert(CarryRecoveryExpert):
    expert_id = "sustained_grasp_to_goal_v1"
    grasp_observations = 6

    def __init__(self, env):
        super().__init__(env)
        self.grasp_history = deque(maxlen=self.grasp_observations)

    def observe(self, info):
        step = int(info["episode"]["steps"])
        previous = self.history[-1]["step"] if self.history else None
        super().observe(info)
        if step == previous:
            return
        if step == 0:
            self.grasp_history.clear()
        self.grasp_history.append(bool(info["metrics"]["grasped"]))

    def eligible(self):
        episode, metrics = self.env._episode, getattr(self.env, "_last_metrics", {})
        if not ("trays" in getattr(self.env, "scene_config", {}) and episode is not None
                and not episode["done"] and episode["ever_grasped"]
                and metrics["grasped"] and not metrics.get("failure_reason")
                and not metrics["dropped_objects"]
                and (not metrics["within_goal"] or not metrics["supported"])):
            return False
        contacts, _, _ = self.env._contacts()
        if any(value["l"] and value["r"] for name, value in contacts.items() if name != self.env.target):
            return False
        return not self._forbidden_contact()

    def should_intervene(self):
        complete = self.history_complete and len(self.grasp_history) == self.grasp_observations
        sustained = complete and all(self.grasp_history)
        self.diagnostics["gate"] = dict(
            kind="proactive_sustained_target_grasp", complete_history=self.history_complete,
            required_grasp_observations=self.grasp_observations,
            sustained_grasp=bool(sustained),
            step=self.history[-1]["step"] if self.history else None,
            requires_prior_lift=False, requires_reference_lag=False,
        )
        return bool(sustained and self.eligible())

    def reset(self, seed=None):
        if not self.eligible():
            raise ValueError("Grasp continuation requires an active target grasp outside a supported correct goal")
        super().reset(seed=seed)
