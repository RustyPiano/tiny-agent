from __future__ import annotations

from dataclasses import dataclass, field

PHASE_IMPLEMENT = "implement"
PHASE_SPEC_REVIEW = "spec_review"
PHASE_QUALITY_REVIEW = "quality_review"
PHASE_TASK_COMPLETE = "task_complete"

STATUS_DONE = "DONE"
STATUS_DONE_WITH_CONCERNS = "DONE_WITH_CONCERNS"
STATUS_NEEDS_CONTEXT = "NEEDS_CONTEXT"
STATUS_BLOCKED = "BLOCKED"

_TRANSITIONS: dict[tuple[str, str], tuple[str, str]] = {
    (PHASE_IMPLEMENT, STATUS_DONE): ("advance_phase", PHASE_SPEC_REVIEW),
    (PHASE_IMPLEMENT, STATUS_DONE_WITH_CONCERNS): ("advance_phase", PHASE_SPEC_REVIEW),
    (PHASE_IMPLEMENT, STATUS_NEEDS_CONTEXT): ("wait_for_context", PHASE_IMPLEMENT),
    (PHASE_IMPLEMENT, STATUS_BLOCKED): ("escalate_blocker", PHASE_IMPLEMENT),
    (PHASE_SPEC_REVIEW, STATUS_DONE): ("advance_phase", PHASE_QUALITY_REVIEW),
    (PHASE_SPEC_REVIEW, STATUS_DONE_WITH_CONCERNS): ("advance_phase", PHASE_QUALITY_REVIEW),
    (PHASE_SPEC_REVIEW, STATUS_NEEDS_CONTEXT): ("wait_for_context", PHASE_SPEC_REVIEW),
    (PHASE_SPEC_REVIEW, STATUS_BLOCKED): ("escalate_blocker", PHASE_SPEC_REVIEW),
    (PHASE_QUALITY_REVIEW, STATUS_DONE): ("mark_task_complete", PHASE_TASK_COMPLETE),
    (PHASE_QUALITY_REVIEW, STATUS_DONE_WITH_CONCERNS): (
        "mark_task_complete",
        PHASE_TASK_COMPLETE,
    ),
    (PHASE_QUALITY_REVIEW, STATUS_NEEDS_CONTEXT): ("wait_for_context", PHASE_QUALITY_REVIEW),
    (PHASE_QUALITY_REVIEW, STATUS_BLOCKED): ("escalate_blocker", PHASE_QUALITY_REVIEW),
}


def advance_flow(phase: str, status: str, message: str = "") -> dict:
    transition = _TRANSITIONS.get((phase, status))
    if transition is None:
        return {
            "ok": False,
            "next_action": "wait_for_context",
            "phase": phase,
            "message": "invalid phase/status transition",
        }
    next_action, next_phase = transition
    return {
        "ok": True,
        "next_action": next_action,
        "phase": next_phase,
        "message": message,
    }


@dataclass
class SubagentFlowState:
    tasks: list[str]
    current_index: int = 0
    phase: str = PHASE_IMPLEMENT
    _last_result: dict = field(default_factory=dict)

    def current_task_id(self) -> str | None:
        if self.current_index < 0 or self.current_index >= len(self.tasks):
            return None
        return self.tasks[self.current_index]

    def handle_payload(self, payload: dict) -> dict:
        task_id = payload.get("task_id")
        phase = payload.get("phase")
        status = payload.get("status")
        details = payload.get("details")
        message = details if isinstance(details, str) else ""

        current_task = self.current_task_id()
        if not isinstance(task_id, str) or not task_id:
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": self.phase,
                "message": "task_id is required",
            }
        if current_task is None:
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": self.phase,
                "message": "no active task",
            }
        if task_id != current_task:
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": self.phase,
                "message": "task_id does not match active task",
            }
        if phase != self.phase:
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": self.phase,
                "message": "phase does not match active phase",
            }

        result = advance_flow(phase=phase, status=status, message=message)
        if not result["ok"]:
            return result

        self.phase = result["phase"]
        self._last_result = result
        if self.phase == PHASE_TASK_COMPLETE and self.current_index < len(self.tasks) - 1:
            self.current_index += 1
            self.phase = PHASE_IMPLEMENT

        return result
