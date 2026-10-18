"""Lifecycle transition and eligibility invariant tests (no indexing engine)."""

from __future__ import annotations

import pytest

from app.models_rag import RagDocumentStatus
from app.rag.lifecycle import (
    LifecycleError,
    assert_transition,
    can_transition,
    document_eligible_for_active_snapshot,
    failed_run_may_replace_active_pointer,
    greenheck_allowed_pages,
    greenheck_excluded_pages,
    requires_eligibility_approval_before_processing,
)


def test_happy_path_transitions():
    assert can_transition(
        RagDocumentStatus.UPLOADED, RagDocumentStatus.VALIDATING
    )
    assert can_transition(
        RagDocumentStatus.VALIDATING, RagDocumentStatus.AWAITING_REVIEW
    )
    assert can_transition(
        RagDocumentStatus.AWAITING_REVIEW, RagDocumentStatus.PROCESSING
    )
    assert can_transition(RagDocumentStatus.PROCESSING, RagDocumentStatus.READY)
    assert can_transition(RagDocumentStatus.READY, RagDocumentStatus.ARCHIVED)


def test_illegal_transition_raises():
    with pytest.raises(LifecycleError):
        assert_transition(RagDocumentStatus.UPLOADED, RagDocumentStatus.READY)
    with pytest.raises(LifecycleError):
        assert_transition(RagDocumentStatus.ARCHIVED, RagDocumentStatus.READY)


def test_failed_can_retry_or_archive():
    assert can_transition(RagDocumentStatus.FAILED, RagDocumentStatus.PROCESSING)
    assert can_transition(RagDocumentStatus.FAILED, RagDocumentStatus.ARCHIVED)


def test_eligibility_required_before_processing():
    assert not requires_eligibility_approval_before_processing(
        eligibility_policy_id=None, eligibility_approved=False
    )
    assert not requires_eligibility_approval_before_processing(
        eligibility_policy_id="greenheck_erv_v1", eligibility_approved=False
    )
    assert requires_eligibility_approval_before_processing(
        eligibility_policy_id="greenheck_erv_v1", eligibility_approved=True
    )


def test_greenheck_page_31_policy_scoped_not_global():
    assert 31 in greenheck_excluded_pages(document_policy_id="greenheck_erv_v1")
    assert greenheck_allowed_pages(document_policy_id="greenheck_erv_v1") == frozenset(
        {15, 29, 33}
    )
    assert greenheck_excluded_pages(document_policy_id="generic_v1") == frozenset()
    assert greenheck_allowed_pages(document_policy_id="generic_v1") is None


def test_archived_not_eligible_for_snapshot():
    assert not document_eligible_for_active_snapshot(
        status=RagDocumentStatus.ARCHIVED,
        eligibility_approved=True,
        eligibility_policy_id="greenheck_erv_v1",
    )
    assert document_eligible_for_active_snapshot(
        status=RagDocumentStatus.READY,
        eligibility_approved=True,
        eligibility_policy_id="greenheck_erv_v1",
    )


def test_failed_run_never_replaces_active_pointer():
    assert failed_run_may_replace_active_pointer() is False
