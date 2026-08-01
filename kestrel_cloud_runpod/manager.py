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
    - RunPodTrainingMixin: LoRA training methods
    - RunPodOllamaMixin: durable private-Ollama capacity leases

    Usage:
        manager = RunPodManager()

        # LoRA Training
        session = await manager.start_training_pod("companion-123")
        job_id = await manager.submit_training_job(session, avatar_data, "companion-123")
        status = await manager.poll_training_status(session, job_id)
        lora_data = await manager.download_lora(session, job_id)

        # Ollama leases use acquire_ollama_lease(request), then explicitly
        # touch_ollama_lease(...) and release_ollama_lease(...).
    """

    def __init__(self, config: dict[str, Any] | None = None, mode: str | None = None):
        """Initialize the RunPod manager."""
        super().__init__(config, mode)
