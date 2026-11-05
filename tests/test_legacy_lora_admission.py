"""Legacy NEW work must refuse before touching any manager/session state.

These sentinels cannot produce a successful session, receipt or resource. The
native scopes below are deliberately revoked/unknown, never positive catalog
or physical authority. Historical lifecycle coverage remains separate.
"""

import asyncio

import pytest
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    execution_commit_outcome,
    execution_denial_scopes,
)

from kestrel_cloud_runpod.manager import RunPodManager
from kestrel_cloud_runpod.models import RunPodLegacyAdmissionRetiredError


class NoEffectsManager(RunPodManager):
    def __init__(self):
        # No config, database, provider, HTTP client, lock or Task is admitted.
        pass

    def __getattr__(self, name):
        raise AssertionError(f"retired new work touched manager dependency: {name}")


class NoEffectsSession:
    def __getattr__(self, name):
        raise AssertionError(f"retired new work touched mutable session: {name}")


OPERATIONS = ("training", "submission", "generation", "inference", "session", "resume")


async def _invoke(manager, operation):
    if operation == "training":
        return await manager.start_training_pod("owned-companion", cleanup_token="old-original")
    if operation == "submission":
        return await manager.submit_training_job(NoEffectsSession(), b"private image", "owned-companion")
    if operation == "generation":
        return await manager.generate_with_lora(NoEffectsSession(), "private prompt", "old-flux-path")
    if operation == "inference":
        return await manager.start_inference_pod("owned-companion")
    if operation == "session":
        return await manager.start_session("image", model_name="old-flux-profile")
    if operation == "resume":
        return await manager.resume_stopped_pod(NoEffectsSession(), NoEffectsSession(), 600)
    raise AssertionError(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_legacy_new_work_refuses_before_any_dependency(operation):
    manager = NoEffectsManager()
    original_tasks = asyncio.all_tasks()
    with pytest.raises(RunPodLegacyAdmissionRetiredError, match="retired"):
        await _invoke(manager, operation)
    assert not vars(manager)
    assert asyncio.all_tasks() == original_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("outcome", ("denied", "unknown"))
async def test_legacy_new_refusal_preserves_original_native_denial(operation, outcome):
    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-retirement-proof")) as scope:
        if outcome == "unknown":
            scope.preserve_commit_uncertainty("unknown")
        else:
            scope.revoke("original invocation retired")
        with pytest.raises(ExecutionAuthorityError) as observed:
            await _invoke(NoEffectsManager(), operation)
        if outcome == "unknown":
            assert execution_commit_outcome(observed.value) == "unknown"
        else:
            assert scope in execution_denial_scopes(observed.value)
