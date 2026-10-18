"""
Document lifecycle transition invariants (Phase 6B.1).

Schema/validation helpers only — no indexing engine.
"""

from __future__ import annotations

from app.models_rag import RagDocumentStatus

# uploaded -> validating -> awaiting_review -> processing -> ready
# any stage may fail; ready -> archived via controlled retirement
ALLOWED_TRANSITIONS: dict[RagDocumentStatus, frozenset[RagDocumentStatus]] = {
    RagDocumentStatus.UPLOADED: frozenset(
        {RagDocumentStatus.VALIDATING, RagDocumentStatus.FAILED}
    ),
    RagDocumentStatus.VALIDATING: frozenset(
        {
            RagDocumentStatus.AWAITING_REVIEW,
            RagDocumentStatus.FAILED,
        }
    ),
    RagDocumentStatus.AWAITING_REVIEW: frozenset(
        {
            RagDocumentStatus.PROCESSING,
            RagDocumentStatus.FAILED,
            RagDocumentStatus.ARCHIVED,
        }
    ),
    RagDocumentStatus.PROCESSING: frozenset(
        {RagDocumentStatus.READY, RagDocumentStatus.FAILED}
    ),
    RagDocumentStatus.READY: frozenset({RagDocumentStatus.ARCHIVED}),
    RagDocumentStatus.FAILED: frozenset(
        {
            RagDocumentStatus.VALIDATING,
            RagDocumentStatus.PROCESSING,
            RagDocumentStatus.ARCHIVED,
        }
    ),
    RagDocumentStatus.ARCHIVED: frozenset(),
}


class LifecycleError(ValueError):
    """Illegal document status transition."""


def can_transition(current: RagDocumentStatus, new: RagDocumentStatus) -> bool:
    return new in ALLOWED_TRANSITIONS.get(current, frozenset())


def assert_transition(current: RagDocumentStatus, new: RagDocumentStatus) -> None:
    if not can_transition(current, new):
        raise LifecycleError(
            f"Illegal document status transition: {current.value} -> {new.value}"
        )


def requires_eligibility_approval_before_processing(
    *,
    eligibility_policy_id: str | None,
    eligibility_approved: bool,
) -> bool:
    """
    Newly uploaded PDFs are not automatically trusted.

    Processing/activation requires an explicit policy id and approval flag.
    """
    return bool(
        eligibility_policy_id
        and eligibility_policy_id.strip()
        and eligibility_approved
    )


def greenheck_excluded_pages(*, document_policy_id: str) -> frozenset[int]:
    """
    Page exclusions are policy-scoped.

    Page 31 is excluded only for the Greenheck ERV policy — not globally.
    """
    if document_policy_id == "greenheck_erv_v1":
        return frozenset({31})
    return frozenset()


def greenheck_allowed_pages(*, document_policy_id: str) -> frozenset[int] | None:
    if document_policy_id == "greenheck_erv_v1":
        return frozenset({15, 29, 33})
    return None


def document_eligible_for_active_snapshot(
    *,
    status: RagDocumentStatus,
    eligibility_approved: bool,
    eligibility_policy_id: str | None,
) -> bool:
    """Archived/failed/unapproved documents must not join a new active snapshot."""
    if status != RagDocumentStatus.READY:
        return False
    if not eligibility_approved:
        return False
    if not eligibility_policy_id or not eligibility_policy_id.strip():
        return False
    return True


def failed_run_may_replace_active_pointer() -> bool:
    """Invariant: a failed indexing run must never replace a healthy active index."""
    return False
