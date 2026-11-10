"""
RunPod Manager - Combined Class.

Combines all RunPod functionality from the mixin classes
into a single manager class.
"""

from typing import Any

from .core import RunPodManagerCore
from .ollama import RunPodOllamaMixin
from .training import RunPodTrainingMixin


class RunPodManager(
    RunPodManagerCore,
    RunPodTrainingMixin,
    RunPodOllamaMixin,
):
    """
    Full RunPod GPU instance manager.

    Combines:
    - RunPodManagerCore: SDK operations, profile loading, session management
    - RunPodTrainingMixin: original LoRA workload/capacity recovery
    - RunPodOllamaMixin: durable private-Ollama capacity leases

    Usage:
        manager = RunPodManager()

        # NEW LoRA work requires the host's authorized catalog workflow.
        # Exact original identities remain usable for historical recovery:
        status = await manager.poll_training_status(original_session, original_job_id)
        lora_data = await manager.download_lora(original_session, original_job_id)
        await manager.release_training_pod(original_cleanup_token)

        # Ollama leases use acquire_ollama_lease(request), then explicitly
        # touch_ollama_lease(...) and release_ollama_lease(...).
    """

    def __init__(self, config: dict[str, Any] | None = None, mode: str | None = None):
        """Initialize the RunPod manager."""
        super().__init__(config, mode)
