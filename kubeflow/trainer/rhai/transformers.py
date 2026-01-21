# Copyright 2024 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformersTrainer for HuggingFace Transformers and TRL with auto-instrumentation."""

from dataclasses import dataclass, field
import inspect
import os
import textwrap
from typing import Callable, Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.constants import PVC_URI_SCHEME, S3_URI_SCHEME
from kubeflow.trainer.types import types


@dataclass
class PeriodicCheckpointConfig:
    """Configuration for periodic checkpointing in Transformers trainers.

    Args:
        save_strategy: Strategy for saving checkpoints ("steps", "epoch", or "no")
        save_steps: Number of steps between checkpoints (required if save_strategy="steps")
        save_total_limit: Maximum number of checkpoints to keep (older ones are deleted)
    """

    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = 3

    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = {"steps", "epoch", "no"}
        if self.save_strategy not in valid_strategies:
            raise ValueError(
                f"save_strategy must be one of {valid_strategies}, got '{self.save_strategy}'"
            )

        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be specified when save_strategy='steps'")

        if self.save_total_limit is not None and self.save_total_limit < 1:
            raise ValueError(f"save_total_limit must be >= 1, got {self.save_total_limit}")


@dataclass
class TransformersTrainer:
    """RHAI trainer for HuggingFace Transformers and TRL with auto-instrumentation.

    Args:
        func: The function that encapsulates the entire model training process.
              Must use transformers.Trainer or trl.SFTTrainer internally.
        func_args: The arguments to pass to the function as kwargs.
        packages_to_install: A list of Python packages to install before running the function.
        pip_index_urls: The PyPI URLs from which to install Python packages.
                       The first URL will be the index-url, and remaining ones are extra-index-urls.
        num_nodes: The number of nodes to use for training.
        resources_per_node: The computing resources to allocate per node.
        env: The environment variables to set in the training nodes.
        enable_progression_tracking: Enable HTTP metrics server. Default: True.
        metrics_port: Port for HTTP metrics server. Default: 28080.
                     Valid range: 1024-65535 (non-privileged ports).
                     Ports 0-1023 are reserved and require root privileges.
                     This range is required for OpenShift restricted SCCs and Kubernetes
                     non-root security policies. Common safe ports: 8080-8999, 28000-29000.
        metrics_poll_interval_seconds: How often controller should poll metrics (seconds).
                                       Default: 30. Range: 5-300 (5s to 5min).
                                       Fast jobs: use 5-10s. Long jobs: use 60-120s.
        enable_jit_checkpoint: Enable just-in-time checkpointing on SIGTERM. Default: False.
                              Automatically enabled when output_dir is provided.
        output_dir: Directory for saving checkpoints. Supports PVC URIs (pvc://<name>/<path>)
                   for automatic volume mounting. When provided, automatically enables JIT
                   checkpointing.
        periodic_checkpoint_config: Optional configuration for periodic checkpointing.
                                   See PeriodicCheckpointConfig for available options.

    Raises:
        ValueError: If metrics_port is not in range 1024-65535.
        ValueError: If metrics_poll_interval_seconds is not in range 5-300.
        ValueError: If func is not callable.
        ValueError: If output_dir uses unsupported URI scheme (only pvc:// is supported).
    """

    # Core training function (same as CustomTrainer)
    func: Callable
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    env: Optional[dict[str, str]] = None

    # Instrumentation features
    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    # Checkpoint configuration
    enable_jit_checkpoint: bool = False
    output_dir: Optional[str] = None
    periodic_checkpoint_config: Optional[PeriodicCheckpointConfig] = None

    def __post_init__(self):
        """Validate configuration after initialization.

        Validation ensures compatibility with:
        - OpenShift restricted Security Context Constraints (SCCs)
        - Kubernetes non-root security policies
        - Standard container best practices
        """
        # Validate func is callable
        if not callable(self.func):
            raise ValueError(
                f"func must be callable, got {type(self.func).__name__}. "
                f"Please provide a training function."
            )

        # Validate metrics_port (must work with OpenShift restricted SCCs)
        if not isinstance(self.metrics_port, int):
            raise ValueError(
                f"metrics_port must be an integer, got {type(self.metrics_port).__name__}"
            )

        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ValueError(
                f"metrics_port must be in range 1024-65535 (non-privileged ports), "
                f"got {self.metrics_port}. Ports 0-1023 are reserved and require root privileges. "
                f"This range (1024-65535) is required for OpenShift restricted SCCs and "
                f"Kubernetes non-root containers."
            )

        # Validate metrics_poll_interval_seconds
        if not isinstance(self.metrics_poll_interval_seconds, int):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer, "
                f"got {type(self.metrics_poll_interval_seconds).__name__}"
            )

        if self.metrics_poll_interval_seconds < 5 or self.metrics_poll_interval_seconds > 300:
            raise ValueError(
                f"metrics_poll_interval_seconds must be in range 5-300, "
                f"got {self.metrics_poll_interval_seconds}"
            )
        # Only allow pvc://, s3://, or paths without URI schemes
        if (
            self.output_dir
            and "://" in self.output_dir
            and not (
                self.output_dir.startswith(PVC_URI_SCHEME)
                or self.output_dir.startswith(S3_URI_SCHEME)
            )
        ):
            raise ValueError(
                f"Unsupported storage URI scheme. "
                f"Currently only '{PVC_URI_SCHEME}' and '{S3_URI_SCHEME}' URIs are supported. "
                f"Supported formats: '{PVC_URI_SCHEME}<pvc-name>/<path>', "
                f"'{S3_URI_SCHEME}<bucket>/<path>', or local filesystem paths."
            )

        # Auto-enable JIT checkpoint if output_dir is provided
        if self.output_dir and not self.enable_jit_checkpoint:
            self.enable_jit_checkpoint = True


def _create_checkpoint_instrumentation(checkpoint_config: dict) -> tuple:
    """
    Checkpoint instrumentation code injected into training pods.
    """
    import os
    import re
    import shutil
    import signal
    import threading
    import time

    import torch
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    class CheckpointManager:
        """Manages async just-in-time checkpointing on SIGTERM signal using CUDA streams."""

        def __init__(self, trainer):
            self.trainer = trainer
            self.checkpoint_requested = False
            self._original_sigterm_handler = None
            self.checkpoint_stream = None
            self.checkpoint_thread = None
            self._in_optimizer_step = False

            # Initialize CUDA stream for async checkpoint operations
            try:
                if torch.cuda.is_available():
                    self.checkpoint_stream = torch.cuda.Stream()
                    print("[Kubeflow] CUDA stream initialized for async checkpointing", flush=True)
            except (ImportError, AttributeError):
                print(
                    "[Kubeflow] CUDA not available, checkpointing will be synchronous", flush=True
                )

        def setup_signal_handler(self):
            """Register SIGTERM signal handler for JIT checkpointing."""
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
            print("[Kubeflow] JIT checkpoint signal handler registered for SIGTERM", flush=True)

        def _sigterm_handler(self, signum, frame):
            """Handle SIGTERM by starting async checkpoint immediately."""
            if self.checkpoint_requested:
                return

            print("[Kubeflow] SIGTERM received, starting async checkpoint", flush=True)
            self.checkpoint_requested = True

            # Start checkpoint thread immediately
            self.checkpoint_thread = threading.Thread(
                target=self._async_checkpoint, daemon=True, name="KubeflowJITCheckpoint"
            )
            self.checkpoint_thread.start()

        def _async_checkpoint(self):
            """Execute checkpoint asynchronously, waiting if in optimizer step."""
            try:
                # Wait if we're currently in optimizer step (unsafe to checkpoint)
                while self._in_optimizer_step:
                    time.sleep(0.5)

                current_step = self.trainer.state.global_step
                print(f"[Kubeflow] Starting JIT checkpoint at step {current_step}", flush=True)

                # Get rank for distributed training. Fall back to True for single-process
                # runs or if accelerate is unavailable.
                try:
                    from accelerate import PartialState

                    is_main_process = PartialState().is_main_process
                except Exception:
                    # accelerate not installed or PartialState unavailable - assume single process
                    is_main_process = True

                output_dir = self.trainer._get_output_dir(trial=None)
                checkpoint_path = os.path.join(
                    output_dir, f"{PREFIX_CHECKPOINT_DIR}-{current_step}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)

                # Create sentinel file to mark incomplete checkpoint (only rank 0)
                sentinel_file = os.path.join(checkpoint_path, CHECKPOINT_INCOMPLETE_MARKER)
                if is_main_process:
                    try:
                        with open(sentinel_file, "w") as f:
                            f.write(f"Checkpoint started at step {current_step}")
                    except Exception as e:
                        print(f"[Kubeflow] Warning: Failed to write sentinel file: {e}")

                # Checkpoint using dedicated CUDA stream
                if self.checkpoint_stream is not None:
                    # Wait for default stream to complete all pending operations
                    self.checkpoint_stream.wait_stream(torch.cuda.default_stream())

                    # Record all model parameters on checkpoint stream to prevent deallocation
                    for param in self.trainer.model.parameters():
                        param.record_stream(self.checkpoint_stream)

                    with torch.cuda.stream(self.checkpoint_stream):
                        self.trainer._save_checkpoint(self.trainer.model, trial=None)
                    self.checkpoint_stream.synchronize()
                else:
                    # Fallback if no CUDA stream
                    self.trainer._save_checkpoint(self.trainer.model, trial=None)

                # Remove sentinel on success (only rank 0)
                if is_main_process and os.path.exists(sentinel_file):
                    try:
                        os.remove(sentinel_file)
                    except Exception as e:
                        print(f"[Kubeflow] Warning: Failed to remove sentinel file: {e}")

                print(f"[Kubeflow] JIT checkpoint completed at step {current_step}", flush=True)

            except Exception as e:
                print(f"[Kubeflow] Failed to save JIT checkpoint: {e}", flush=True)
                import traceback

                traceback.print_exc()

        def checkpoint_in_progress(self):
            """Check if a checkpoint is in progress."""
            return self.checkpoint_requested

    class JITCheckpointCallback(TrainerCallback):
        """Transformers callback that integrates JIT checkpointing with trainer lifecycle."""

        def __init__(self, storage_uri=None):
            self.jit_manager = None
            self._trainer_ref = None

            # Storage configuration (supports s3://, gs://, az://, etc.)
            self.storage_uri = storage_uri
            self.storage_fs = None
            self.storage_protocol = None
            self.storage_base_path = None  # Base path in storage (bucket/prefix)

            if storage_uri and "://" in storage_uri:
                import subprocess
                import sys

                # Pin fsspec version to avoid conflicts with datasets package
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "s3fs", "fsspec<=2025.3.0"]
                )
                # Parse storage URI: protocol://bucket/prefix
                self.storage_protocol = storage_uri.split("://")[0]
                path_part = storage_uri.split("://", 1)[1]

                # Storage base path is everything after protocol (bucket/prefix)
                self.storage_base_path = path_part

                # Initialize fsspec filesystem (auto-selects backend: s3fs, gcsfs, adlfs, etc.)
                import fsspec

                # Protocol-specific configuration (only non-standard settings)
                # fsspec backends auto-detect credentials from environment variables
                fs_kwargs = {}
                if self.storage_protocol == "s3":
                    # Only pass custom endpoint and SSL settings for non-AWS S3 (MinIO, Ceph, etc.)
                    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
                    if endpoint_url:
                        fs_kwargs = {
                            "client_kwargs": {
                                "endpoint_url": endpoint_url,
                                "verify": False,  # For self-signed certs
                            },
                            "config_kwargs": {"signature_version": "s3v4"},
                        }
                # GCS and Azure auto-detect credentials, no custom config needed

                self.storage_fs = fsspec.filesystem(self.storage_protocol, **fs_kwargs)
                print(
                    f"[Kubeflow] Storage configured: {storage_uri} ({self.storage_protocol})",
                    flush=True,
                )

        def on_train_begin(self, args, state, control, **kwargs):
            if self._trainer_ref is not None and self.jit_manager is None:
                self.jit_manager = CheckpointManager(trainer=self._trainer_ref)
                self.jit_manager.setup_signal_handler()
                print("[Kubeflow] JIT checkpointing enabled", flush=True)
            elif self._trainer_ref is None:
                print(
                    "[Kubeflow] Warning: Trainer reference not set for JIT checkpoint callback",
                    flush=True,
                )

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager:
                # Mark that we're entering optimizer step (unsafe for checkpoint)
                self.jit_manager._in_optimizer_step = True

                if self.jit_manager.checkpoint_in_progress():
                    control.should_training_stop = True

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager:
                # Mark that optimizer step completed (safe for checkpoint again)
                self.jit_manager._in_optimizer_step = False

        def on_step_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                control.should_save = False
                control.should_training_stop = True

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                control.should_save = False
                control.should_training_stop = True

        def _should_upload_download(self, args, state):
            """Determine if this rank should upload/download checkpoints.

            Returns True if:
            - FSDP with SHARDED_STATE_DICT: All ranks participate (parallel upload/download)
            - FSDP with FULL_STATE_DICT: Only rank 0
            - DeepSpeed ZeRO-3: All ranks participate (placeholder for future)
            - Otherwise: Only rank 0
            """
            # Check for FSDP SHARDED_STATE_DICT
            if (
                hasattr(args, "fsdp")
                and args.fsdp
                and hasattr(args, "fsdp_config")
                and args.fsdp_config
            ):
                state_dict_type = args.fsdp_config.get("state_dict_type", "FULL_STATE_DICT")
                if state_dict_type == "SHARDED_STATE_DICT":
                    # All ranks participate in parallel upload/download
                    return True

            # Check for DeepSpeed (placeholder for future implementation)
            if hasattr(args, "deepspeed") and args.deepspeed:
                # TODO: Detect DeepSpeed ZeRO-3 and return True for parallel upload/download
                # For now, fall back to rank 0 only
                pass

            # Default: Only rank 0 uploads/downloads
            return state.is_world_process_zero

        def on_init_end(self, args, state, control, **kwargs):
            """Download latest checkpoint from storage before training starts.

            Handles distributed training correctly:
            - FSDP SHARDED_STATE_DICT: All ranks download in parallel
              (each rank gets full checkpoint)
            - FSDP FULL_STATE_DICT: Rank 0 downloads, all ranks wait at barrier
            - DeepSpeed: Placeholder for future implementation
            """
            if not self.storage_fs:
                return

            # Get rank number for logging
            rank = 0
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass

            try:
                # Determine if this rank should download
                should_download = self._should_upload_download(args, state)
                is_rank_0 = state.is_world_process_zero

                # Check if we need to download (checkpoint doesn't exist locally)
                latest_checkpoint_name = None
                local_checkpoint_exists = False

                # For FSDP SHARDED_STATE_DICT: All ranks participate
                # For others: Only rank 0 downloads
                if should_download or is_rank_0:
                    print(
                        f"[Rank {rank}] Checking for checkpoints in {self.storage_protocol}...",
                        flush=True,
                    )

                    # Find latest valid checkpoint in storage
                    latest_checkpoint_name = self._find_latest_checkpoint_storage()
                    if not latest_checkpoint_name:
                        print(
                            f"[Rank {rank}] No checkpoint found in storage, "
                            f"starting fresh training",
                            flush=True,
                        )
                        # Use barrier to ensure all ranks know there's no checkpoint
                        try:
                            import torch.distributed as dist

                            if dist.is_initialized():
                                dist.barrier()
                        except Exception:
                            pass
                        return

                    # Check if checkpoint already exists locally (from previous run)
                    local_checkpoint_path = os.path.join(args.output_dir, latest_checkpoint_name)
                    local_checkpoint_exists = os.path.exists(local_checkpoint_path)

                    if local_checkpoint_exists:
                        print(
                            f"[Rank {rank}] Checkpoint already exists "
                            f"locally: {latest_checkpoint_name}",
                            flush=True,
                        )
                    else:
                        # Download to local output_dir
                        # FSDP SHARDED_STATE_DICT: Each rank downloads full checkpoint
                        # to its ephemeral volume
                        # FSDP FULL_STATE_DICT: Only rank 0 downloads to shared volume
                        print(
                            f"[Rank {rank}] Downloading checkpoint: "
                            f"{latest_checkpoint_name} to {args.output_dir}",
                            flush=True,
                        )
                        start_time = time.time()

                        # Pass rank for selective download in SHARDED_STATE_DICT mode
                        download_rank = rank if should_download else None
                        self._download_checkpoint_from_storage(
                            latest_checkpoint_name, args.output_dir, rank=download_rank
                        )

                        duration = time.time() - start_time
                        print(
                            f"[Rank {rank}] ✓ Download complete: "
                            f"{latest_checkpoint_name} in {duration:.2f}s",
                            flush=True,
                        )

                # CRITICAL: Use distributed barrier to ensure all
                # ranks wait for downloads to finish
                # For FSDP SHARDED_STATE_DICT: Wait for all ranks to finish downloading
                # For FSDP FULL_STATE_DICT: Wait for rank 0 to finish downloading
                try:
                    import torch.distributed as dist

                    if dist.is_initialized():
                        # All ranks wait here until downloads complete
                        print(
                            f"[Rank {rank}] Waiting at barrier for checkpoint sync...", flush=True
                        )
                        dist.barrier()
                        print(
                            f"[Rank {rank}] Barrier passed, checkpoint ready for loading",
                            flush=True,
                        )
                except Exception as e:
                    # If barrier fails (e.g., single process run), just continue
                    print(
                        f"[Rank {rank}] Warning: Distributed barrier not available: {e}", flush=True
                    )

            except Exception as e:
                print(f"[Rank {rank}] Warning: Failed during checkpoint download: {e}", flush=True)
                import traceback

                traceback.print_exc()

        def on_save(self, args, state, control, **kwargs):
            """Upload checkpoint to storage after HuggingFace saves it locally.

            Handles distributed training correctly:
            - FSDP SHARDED_STATE_DICT: All ranks upload in parallel
              (each uploads full checkpoint dir)
            - FSDP FULL_STATE_DICT: Only rank 0 uploads
            - DeepSpeed: Placeholder for future implementation
            """
            if not self.storage_fs:
                return

            # Determine if this rank should upload
            should_upload = self._should_upload_download(args, state)
            if not should_upload:
                return

            # Get rank number for logging
            rank = 0
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass

            try:
                # Get the checkpoint that was just saved
                checkpoint_name = f"checkpoint-{state.global_step}"
                local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

                if not os.path.exists(local_checkpoint_path):
                    print(
                        f"[Rank {rank}] Warning: Checkpoint not found: {local_checkpoint_path}",
                        flush=True,
                    )
                    return

                print(
                    f"[Rank {rank}] Uploading checkpoint to {self.storage_protocol}: "
                    f"{checkpoint_name}",
                    flush=True,
                )
                start_time = time.time()

                # Upload checkpoint to storage (synchronous)
                # For FSDP SHARDED_STATE_DICT: Each rank uploads only its files (selective)
                # For FSDP FULL_STATE_DICT: Only rank 0 uploads (all files)
                upload_rank = rank if should_upload else None
                is_rank_0 = state.is_world_process_zero
                self._upload_checkpoint_to_storage(
                    local_checkpoint_path, checkpoint_name, rank=upload_rank, is_rank_0=is_rank_0
                )

                duration = time.time() - start_time
                size_mb = self._get_dir_size_mb(local_checkpoint_path)
                throughput = size_mb / duration if duration > 0 else 0
                print(
                    f"[Rank {rank}] ✓ Upload complete: {checkpoint_name} "
                    f"({size_mb:.1f}MB in {duration:.2f}s, {throughput:.2f} MB/s)",
                    flush=True,
                )

            except Exception as e:
                print(f"[Rank {rank}] Error uploading to storage: {e}", flush=True)
                import traceback

                traceback.print_exc()

        def _find_latest_checkpoint_storage(self):
            """Find latest valid checkpoint in storage bucket.

            Returns:
                str: Checkpoint name (e.g., 'checkpoint-100') or None if not found
            """
            try:
                # List all checkpoint directories using glob pattern
                checkpoint_dirs = self.storage_fs.glob(f"{self.storage_base_path}/checkpoint-*")

                # Extract step numbers and sort descending
                checkpoint_steps = sorted(
                    [int(path.split("-")[-1]) for path in checkpoint_dirs],
                    reverse=True,
                )

                if not checkpoint_steps:
                    return None

                # Find first complete checkpoint (latest to oldest)
                for step in checkpoint_steps:
                    checkpoint_name = f"checkpoint-{step}"
                    incomplete_marker = (
                        f"{self.storage_base_path}/{checkpoint_name}/{CHECKPOINT_INCOMPLETE_MARKER}"
                    )

                    if self.storage_fs.exists(incomplete_marker):
                        print(f"[Kubeflow] Skipping incomplete: {checkpoint_name}", flush=True)
                        continue

                    print(
                        f"[Kubeflow] Found {len(checkpoint_steps)} checkpoints, "
                        f"using: {checkpoint_name}",
                        flush=True,
                    )
                    return checkpoint_name

                print("[Kubeflow] All checkpoints incomplete", flush=True)
                return None

            except Exception as e:
                print(f"[Kubeflow] Error finding checkpoint in storage: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return None

        def _is_rank_specific_file(self, file_path, rank):
            """Check if a file is rank-specific for FSDP SHARDED_STATE_DICT.

            Rank-specific files:
            - Model/optimizer shards: __<rank>_0.distcp
            - RNG state: rng_state_<rank>.pth

            Shared files (all ranks need):
            - .metadata files
            - scheduler.pt
            - trainer_state.json
            - config.json
            - tokenizer files, etc.
            """
            filename = os.path.basename(file_path)

            # Pattern 1: Shard files like __0_0.distcp, __1_0.distcp
            if f"__{rank}_0.distcp" in filename:
                return True

            # Pattern 2: RNG state files like rng_state_0.pth, rng_state_1.pth
            return filename == f"rng_state_{rank}.pth"

        def _is_shared_file(self, file_path):
            """Check if a file is shared across all ranks (not rank-specific).

            Shared files include:
            - .metadata files
            - scheduler.pt
            - trainer_state.json
            - config files
            - tokenizer files
            - Any file that doesn't match rank-specific patterns
            """
            filename = os.path.basename(file_path)

            # Shared files patterns
            shared_patterns = [
                ".metadata",
                "scheduler.pt",
                "trainer_state.json",
                "config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
            ]

            # Check if it matches any shared pattern
            for pattern in shared_patterns:
                if pattern in filename:
                    return True

            # If it doesn't match rank-specific patterns, it's shared
            # (rank-specific: __<N>_0.distcp or rng_state_<N>.pth)
            import re

            return not (
                re.match(r".*__\d+_0\.distcp$", filename)
                or re.match(r"rng_state_\d+\.pth$", filename)
            )

        def _download_checkpoint_from_storage(self, checkpoint_name, local_dir, rank=None):
            """Download checkpoint from storage to local directory.

            For FSDP SHARDED_STATE_DICT:
            - Rank 0: Downloads shared files + ALL shard files
            - Other ranks: Download ALL shard files only (no shared files)
            - All ranks need all shards for PyTorch distributed checkpoint loader

            For FSDP FULL_STATE_DICT or single-process:
            - Downloads entire checkpoint

            Args:
                checkpoint_name: Name of checkpoint (e.g., 'checkpoint-100')
                local_dir: Local directory to download to (args.output_dir)
                rank: Rank number for selective download (None = download all)
            """
            storage_path = f"{self.storage_base_path}/{checkpoint_name}"

            os.makedirs(local_dir, exist_ok=True)

            if rank is None:
                # Download all files (FULL_STATE_DICT or single process)
                print(
                    f"[Kubeflow] Downloading from {self.storage_protocol}://{storage_path} "
                    f"to {local_dir}",
                    flush=True,
                )
                self.storage_fs.get(storage_path, local_dir, recursive=True)
            else:
                # Selective download for SHARDED_STATE_DICT
                # List all files in the checkpoint
                all_files = self.storage_fs.find(storage_path)

                files_to_download = []
                is_rank_0 = rank == 0

                for remote_file in all_files:
                    # Get relative path within checkpoint
                    rel_path = remote_file.replace(f"{storage_path}/", "")

                    is_shared = self._is_shared_file(rel_path)
                    is_my_shard = self._is_rank_specific_file(rel_path, rank)

                    if is_rank_0:
                        # Rank 0: download everything (orchestrates checkpoint loading)
                        files_to_download.append(remote_file)
                    else:
                        # Other ranks: download shared files + only their own shard
                        if is_shared or is_my_shard:
                            files_to_download.append(remote_file)

                download_desc = "all files" if is_rank_0 else f"shared + rank-{rank} shard"
                print(
                    f"[Rank {rank}] Downloading {len(files_to_download)} files "
                    f"({download_desc}) from {self.storage_protocol}://{storage_path}",
                    flush=True,
                )

                # Download each file
                for remote_file in files_to_download:
                    # Get relative path and construct local path
                    rel_path = remote_file.replace(f"{storage_path}/", "")
                    local_file = os.path.join(local_dir, checkpoint_name, rel_path)

                    # Create parent directory
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    # Download file
                    self.storage_fs.get(remote_file, local_file)

        def _upload_checkpoint_to_storage(
            self, local_checkpoint_path, checkpoint_name, rank=None, is_rank_0=False
        ):
            """Upload checkpoint from local directory to storage.

            For FSDP SHARDED_STATE_DICT:
            - Rank 0: Uploads shared files + its own shards
            - Other ranks: Upload only their own shards

            For FSDP FULL_STATE_DICT or single-process:
            - Uploads entire checkpoint

            Args:
                local_checkpoint_path: Local path to checkpoint directory
                checkpoint_name: Name of checkpoint (e.g., 'checkpoint-100')
                rank: Rank number for selective upload (None = upload all)
                is_rank_0: Whether this is rank 0 (for incomplete marker handling)
            """
            storage_path = f"{self.storage_base_path}"

            # Create incomplete marker to indicate upload in progress (only rank 0)
            incomplete_marker = f"{storage_path}/{checkpoint_name}/{CHECKPOINT_INCOMPLETE_MARKER}"
            if is_rank_0:
                self.storage_fs.touch(incomplete_marker)
                print(f"[Rank 0] Created incomplete marker: {incomplete_marker}", flush=True)

            try:
                if rank is None:
                    # Upload all files (FULL_STATE_DICT or single process)
                    print(
                        f"[Kubeflow] Uploading {local_checkpoint_path} to {self.storage_protocol}://{storage_path}",
                        flush=True,
                    )
                    self.storage_fs.put(local_checkpoint_path, storage_path, recursive=True)
                else:
                    # Selective upload for SHARDED_STATE_DICT
                    # Walk local directory and find files to upload
                    files_to_upload = []
                    for dirpath, _, filenames in os.walk(local_checkpoint_path):
                        for filename in filenames:
                            local_file = os.path.join(dirpath, filename)
                            # Get relative path within checkpoint
                            rel_path = os.path.relpath(local_file, local_checkpoint_path)

                            # Upload if:
                            # - Rank 0: shared files OR rank 0 specific files
                            # - Other ranks: only their own rank-specific files
                            if is_rank_0:
                                # Rank 0 uploads shared files AND its own shards
                                if self._is_shared_file(rel_path) or self._is_rank_specific_file(
                                    rel_path, rank
                                ):
                                    files_to_upload.append((local_file, rel_path))
                            else:
                                # Other ranks upload only their own shards
                                if self._is_rank_specific_file(rel_path, rank):
                                    files_to_upload.append((local_file, rel_path))

                    shard_desc = f"shared + rank-{rank}" if is_rank_0 else f"rank-{rank}"
                    print(
                        f"[Rank {rank}] Uploading {len(files_to_upload)} files "
                        f"({shard_desc} shards) "
                        f"to {self.storage_protocol}://{storage_path}/{checkpoint_name}",
                        flush=True,
                    )

                    # Upload each file
                    for local_file, rel_path in files_to_upload:
                        remote_file = f"{storage_path}/{checkpoint_name}/{rel_path}"
                        self.storage_fs.put(local_file, remote_file)

                # Delete incomplete marker on success (only rank 0)
                if is_rank_0:
                    try:
                        if self.storage_fs.exists(incomplete_marker):
                            # Access underlying boto3 client for single file delete
                            self.storage_fs.rm_file(incomplete_marker)
                            print("[Rank 0] Removed incomplete marker", flush=True)
                    except Exception as delete_error:
                        print(
                            f"[Rank 0] Warning: Could not delete incomplete marker: {delete_error}",
                            flush=True,
                        )

            except Exception as e:
                # Keep incomplete marker if upload fails
                rank_msg = f"[Rank {rank}]" if rank is not None else "[Kubeflow]"
                print(
                    f"{rank_msg} Upload failed due to: {e}, keeping incomplete marker", flush=True
                )
                raise

        def _get_dir_size_mb(self, directory):
            """Get total size of directory in MB."""
            total_size = 0
            for dirpath, _, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):  # Handle broken symlinks
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)

    def apply_checkpointing():
        """Setup monkey patch for Trainer to auto inject JIT checkpoint callback."""
        from transformers import Trainer as _TransformersTrainer

        # Get storage URI from checkpoint config (supports s3://, gs://, az://, etc.)
        storage_uri = checkpoint_config.get("storage_uri")

        _jit_checkpoint_callback = JITCheckpointCallback(storage_uri=storage_uri)

        def _find_latest_checkpoint(output_dir):
            """Find the latest checkpoint and deleting incomplete ones."""
            if not output_dir or not os.path.exists(output_dir):
                return None

            # Determine if this is rank 0 (main process). Fall back to True for single-process
            # runs or if accelerate is unavailable.
            try:
                from accelerate import PartialState

                is_rank_0 = PartialState().is_main_process
            except Exception:
                # accelerate not installed or PartialState unavailable - assume single process
                is_rank_0 = True

            checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
            checkpoints = []

            for name in os.listdir(output_dir):
                match = checkpoint_pattern.match(name)
                if not match or not os.path.isdir(os.path.join(output_dir, name)):
                    continue

                checkpoint_path = os.path.join(output_dir, name)
                incomplete_marker = os.path.join(checkpoint_path, CHECKPOINT_INCOMPLETE_MARKER)

                # Delete incomplete checkpoints (rank 0 only to avoid race condition)
                if os.path.exists(incomplete_marker):
                    if is_rank_0:
                        try:
                            print(f"[Kubeflow] Deleting incomplete checkpoint: {checkpoint_path}")
                            shutil.rmtree(checkpoint_path)
                        except Exception as e:
                            print(f"[Kubeflow] Warning: Failed to delete checkpoint: {e}")
                    continue

                checkpoints.append((int(match.group(1)), checkpoint_path))

            if checkpoints:
                checkpoints.sort(reverse=True)
                latest = checkpoints[0][1]
                print(f"[Kubeflow] Found latest checkpoint: {latest}")
                return latest

            return None

        # Store original __init__ method
        _original_trainer_init = _TransformersTrainer.__init__

        def _patched_trainer_init(self, *args, **kwargs):
            """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
            enable_jit = checkpoint_config.get("enable_jit", False)

            # Extract TrainingArguments to patch
            training_args = kwargs.get("args")
            if not training_args and len(args) > 1:
                training_args = args[1]

            # Apply Kubeflow checkpoint config to training_args
            if training_args and checkpoint_config:
                # Apply output_dir if provided by user
                if "output_dir" in checkpoint_config:
                    training_args.output_dir = checkpoint_config["output_dir"]
                    print(
                        f"[Kubeflow] Applied output_dir: {checkpoint_config['output_dir']}",
                        flush=True,
                    )

                if "save_strategy" in checkpoint_config:
                    training_args.save_strategy = checkpoint_config["save_strategy"]
                    print(
                        f"[Kubeflow] Applied save_strategy: {checkpoint_config['save_strategy']}",
                        flush=True,
                    )

                if (
                    "save_steps" in checkpoint_config
                    and checkpoint_config["save_steps"] is not None
                ):
                    training_args.save_steps = checkpoint_config["save_steps"]
                    print(
                        f"[Kubeflow] Applied save_steps: {checkpoint_config['save_steps']}",
                        flush=True,
                    )

                if "save_total_limit" in checkpoint_config:
                    training_args.save_total_limit = checkpoint_config["save_total_limit"]
                    print(
                        f"[Kubeflow] Applied save_total_limit: "
                        f"{checkpoint_config['save_total_limit']}",
                        flush=True,
                    )

            # Inject JIT callback if enabled
            if enable_jit:
                callbacks = kwargs.get("callbacks") or []
                if not isinstance(callbacks, list):
                    callbacks = list(callbacks)
                if not any(isinstance(cb, JITCheckpointCallback) for cb in callbacks):
                    callbacks.append(_jit_checkpoint_callback)
                    print("[Kubeflow] Auto-injected JIT checkpoint callback", flush=True)
                kwargs["callbacks"] = callbacks

            # Call original __init__
            _original_trainer_init(self, *args, **kwargs)

            # Store trainer reference in callback
            if enable_jit:
                _jit_checkpoint_callback._trainer_ref = self

            _original_train = self.train

            def _patched_train(resume_from_checkpoint=None, **train_kwargs):
                """Patched train() that auto-resumes from latest checkpoint if available."""

                # Only auto-resume if user didn't explicitly set it
                if resume_from_checkpoint is None and training_args:
                    latest_checkpoint = _find_latest_checkpoint(training_args.output_dir)
                    if latest_checkpoint:
                        resume_from_checkpoint = latest_checkpoint
                        print(f"[Kubeflow] Auto-resuming from: {latest_checkpoint}")
                return _original_train(
                    resume_from_checkpoint=resume_from_checkpoint, **train_kwargs
                )

            self.train = _patched_train

        # Apply monkey-patch
        _TransformersTrainer.__init__ = _patched_trainer_init
        print("[Kubeflow] Trainer auto-instrumentation enabled", flush=True)

    enable_jit = checkpoint_config.get("enable_jit", False)
    return (
        CheckpointManager if enable_jit else None,
        JITCheckpointCallback if enable_jit else None,
        apply_checkpointing,
    )


def _create_progression_instrumentation(metrics_port: int) -> tuple:
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into user training scripts. This approach
    provides syntax highlighting, testability, and type checking while avoiding
    runtime SDK dependencies.

    Args:
        metrics_port: Port for HTTP metrics server

    Returns:
        Tuple of (apply_fn, callback_class, handler_class) for testing purposes
    """
    from dataclasses import asdict, dataclass, field
    import http.server
    import json
    import threading
    import time
    from typing import Any, Optional

    from transformers import TrainerCallback, trainer as trainer_module

    @dataclass
    class ProgressionMetricsState:
        """Progression metrics state (camelCase for Kubernetes API compatibility)."""

        progressPercentage: Optional[int] = None  # noqa: N815
        estimatedRemainingSeconds: Optional[int] = None  # noqa: N815
        currentStep: int = 0  # noqa: N815
        totalSteps: Optional[int] = None  # noqa: N815
        currentEpoch: float = 0.0  # noqa: N815  # Changed to float for precision (1.98 not 1)
        totalEpochs: Optional[int] = None  # noqa: N815
        trainMetrics: dict[str, Any] = field(default_factory=dict)  # noqa: N815
        evalMetrics: dict[str, Any] = field(default_factory=dict)  # noqa: N815

    _progression_metrics_state = ProgressionMetricsState()
    _progression_metrics_lock = threading.Lock()

    def _update_progression_metrics(updates: dict) -> None:
        """Thread-safe metrics update."""
        with _progression_metrics_lock:
            for key, value in updates.items():
                if hasattr(_progression_metrics_state, key):
                    current_value = getattr(_progression_metrics_state, key)
                    if isinstance(value, dict) and isinstance(current_value, dict):
                        current_value.update(value)
                    else:
                        setattr(_progression_metrics_state, key, value)

    def _get_progression_metrics_json() -> str:
        """Get current metrics as JSON string."""
        with _progression_metrics_lock:
            return json.dumps(asdict(_progression_metrics_state))

    class ProgressionMetricsHandler(http.server.BaseHTTPRequestHandler):
        """HTTP server that exposes training progress metrics as JSON."""

        def log_message(self, format, *args):
            """Suppress HTTP server logging."""
            pass

        def do_GET(self):
            """Handle GET requests to expose metrics as JSON."""
            try:
                payload = _get_progression_metrics_json()
            except Exception as e:
                print(f"[Kubeflow] Failed to create progress metrics payload: {e}")
                self.send_error(500)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))

    class KubeflowProgressCallback(TrainerCallback):
        """Tracks training progress and updates metrics server."""

        def __init__(self, metrics_port: int = 28080):
            self.start_time: Optional[float] = None
            self.metrics_port = metrics_port
            self.server: Optional[http.server.HTTPServer] = None
            self.training_finished: bool = False

        def _update_progress_state(self, args, state) -> None:
            """Calculate and update progression state during training."""
            current_step = state.global_step if state.global_step is not None else 0
            total_steps = state.max_steps

            # Calculate progress percentage (always rounds down, e.g., 374/375 = 99%)
            progress_pct = int(current_step / total_steps * 100) if total_steps > 0 else 0

            # Estimate remaining time based on elapsed time and progress
            current_time = time.time()
            elapsed_sec = current_time - self.start_time if self.start_time else 0
            remaining_sec = None
            if total_steps > 0 and current_step > 0 and elapsed_sec > 0:
                # If training reached completion, set remaining time to 0
                if current_step >= total_steps:
                    remaining_sec = 0
                else:
                    estimated_total_time = elapsed_sec / (current_step / total_steps)
                    remaining_sec = max(0, int(estimated_total_time - elapsed_sec))

            # Calculate current epoch (keep float precision, e.g., 1.98)
            current_epoch = 0.0
            if hasattr(state, "epoch") and state.epoch:
                current_epoch = float(state.epoch)

            _update_progression_metrics(
                {
                    "currentStep": current_step,
                    "totalSteps": total_steps if total_steps > 0 else None,
                    "currentEpoch": current_epoch,
                    "progressPercentage": progress_pct,
                    "estimatedRemainingSeconds": remaining_sec,
                }
            )

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            """Initialize progress tracking when training starts."""
            self.start_time = time.time()

            if state.is_world_process_zero and self.server is None:
                try:
                    server = http.server.HTTPServer(
                        ("0.0.0.0", self.metrics_port), ProgressionMetricsHandler
                    )
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    self.server = server
                    print(
                        f"[Kubeflow] Metrics server started on port {self.metrics_port}",
                        flush=True,
                    )
                except OSError as e:
                    print(
                        f"[Kubeflow] Warning: Failed to start metrics server on port "
                        f"{self.metrics_port}: {e}. Training will continue without metrics server.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                        f"Training will continue without metrics server.",
                        flush=True,
                    )

            # Calculate initial progress (handles checkpoint resume scenarios)
            initial_progress = 0
            current_step = state.global_step if state.global_step is not None else 0
            if state.max_steps > 0 and current_step > 0:
                initial_progress = int(current_step / state.max_steps * 100)

            _update_progression_metrics(
                {
                    "currentStep": current_step,
                    "totalSteps": state.max_steps if state.max_steps > 0 else None,
                    "currentEpoch": (
                        float(state.epoch) if hasattr(state, "epoch") and state.epoch else 0.0
                    ),
                    "totalEpochs": (
                        int(args.num_train_epochs) if hasattr(args, "num_train_epochs") else None
                    ),
                    "progressPercentage": initial_progress,
                }
            )

        def on_step_end(self, args, state, control, **kwargs) -> None:
            """Update progress after each training step."""
            # Don't overwrite completion state if training has already ended
            if not self.training_finished:
                self._update_progress_state(args, state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Categorize and track training/evaluation metrics."""
            if logs:
                train_metrics = {}
                eval_metrics = {}

                for key, value in logs.items():
                    metric_value = None
                    if isinstance(value, (int, float)):
                        metric_value = value
                    else:
                        # Try to extract scalar from torch tensor (graceful degradation)
                        try:
                            import torch  # type: ignore[import-not-found]

                            if isinstance(value, torch.Tensor) and value.numel() == 1:
                                metric_value = value.item()
                        except (ImportError, AttributeError):
                            pass

                    if metric_value is None:
                        continue

                    if key.startswith("eval_"):
                        eval_metrics[key] = metric_value
                    elif key in ["loss", "learning_rate", "grad_norm", "train_loss"]:
                        train_metrics[key] = metric_value
                    elif key == "train_samples_per_second":
                        train_metrics["throughput_samples_sec"] = metric_value

                update_dict = {}
                if train_metrics:
                    update_dict["trainMetrics"] = train_metrics
                if eval_metrics:
                    update_dict["evalMetrics"] = eval_metrics

                if update_dict:
                    _update_progression_metrics(update_dict)

        def on_train_end(self, args, state, control, **kwargs) -> None:
            """Update final progression state and write to termination message."""
            self.training_finished = True

            total_steps = state.max_steps if state.max_steps > 0 else None
            total_epochs = int(args.num_train_epochs) if hasattr(args, "num_train_epochs") else None

            # Calculate actual progress percentage (with safety checks)
            current_step = state.global_step if state.global_step is not None else 0
            progress_pct = (
                int(current_step / total_steps * 100)
                if total_steps and total_steps > 0 and current_step >= 0
                else 100
            )

            final_metrics = {
                "currentStep": current_step,
                "totalSteps": total_steps,
                "currentEpoch": (
                    float(state.epoch) if hasattr(state, "epoch") and state.epoch else 0.0
                ),
                "totalEpochs": total_epochs,
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": 0,
            }

            # Update HTTP server metrics
            _update_progression_metrics(final_metrics)

            # Write final metrics to termination message for controller capture
            if state.is_world_process_zero:
                try:
                    import json

                    # Hold lock during message construction and file write
                    # (to prevent race conditions)
                    with _progression_metrics_lock:
                        termination_message = {
                            "progressPercentage": progress_pct,
                            "estimatedRemainingSeconds": 0,
                            "currentStep": current_step,
                            "totalSteps": total_steps,
                            "currentEpoch": (
                                float(state.epoch)
                                if hasattr(state, "epoch") and state.epoch
                                else 0.0
                            ),
                            "totalEpochs": total_epochs,
                            "trainMetrics": dict(_progression_metrics_state.trainMetrics),
                            "evalMetrics": dict(_progression_metrics_state.evalMetrics),
                        }

                        with open("/dev/termination-log", "w") as f:
                            json.dump(termination_message, f)
                    print("[Kubeflow] Final metrics written to termination message", flush=True)
                except Exception as e:
                    print(
                        f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                        f"Controller will fall back to HTTP polling.",
                        flush=True,
                    )

    def apply_progression_tracking():
        """Patch Trainer.__init__ to inject KubeflowProgressCallback."""
        _original_init = trainer_module.Trainer.__init__

        def _instrumented_trainer_init(self, *args, **kwargs):
            result = _original_init(self, *args, **kwargs)
            callback = KubeflowProgressCallback(metrics_port)
            if callback not in self.callback_handler.callbacks:
                self.add_callback(callback)
            return result

        trainer_module.Trainer.__init__ = _instrumented_trainer_init

    # Return callback class and helper functions (helpers exposed for testing)
    return (
        apply_progression_tracking,
        KubeflowProgressCallback,
        ProgressionMetricsHandler,
        _get_progression_metrics_json,
        _update_progression_metrics,
    )


def get_transformers_instrumentation_wrapper(
    metrics_port: int,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Extracts _create_progression_instrumentation as source code and injects a call
    with the provided metrics_port parameter.

    Args:
        metrics_port: Port for HTTP metrics server.

    Returns:
        Python code as string with {{user_func_import_and_call}} placeholder.
    """
    import inspect
    import textwrap

    # Extract the entire function source
    progression_instrumentation_code = inspect.getsource(_create_progression_instrumentation)
    progression_instrumentation_code = textwrap.dedent(progression_instrumentation_code)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.transformers
# =============================================================================

print("[Kubeflow] Initializing progression tracking", flush=True)

# Instrumentation function definition
{progression_instrumentation_code}

# Initialize and apply instrumentation
(
    apply_progression_tracking,
    _,
    _,
    _,
    _,
) = _create_progression_instrumentation(metrics_port={metrics_port})
apply_progression_tracking()
print("[Kubeflow] Progression tracking enabled", flush=True)

# =============================================================================
# USER TRAINING CODE
# =============================================================================

{{{{user_func_import_and_call}}}}"""

    return wrapper


def get_trainer_cr_from_transformers_trainer(
    runtime: types.Runtime,
    trainer: TransformersTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for TransformersTrainer with optional progression tracking.

    Args:
        runtime: Training runtime configuration
        trainer: TransformersTrainer instance
        initializer: Optional dataset/model initializer

    Returns:
        Trainer CRD with wrapped training function and annotations
    """
    from kubeflow.trainer.backends.kubernetes import utils
    from kubeflow.trainer.constants import constants

    # Ensure runtime trainer has a command set
    # This handles cases where RuntimeTrainer is created without calling set_command()
    try:
        _ = runtime.trainer.command
    except AttributeError:
        # Command not set, use default based on framework
        if runtime.trainer.framework == "pytorch":
            runtime.trainer.set_command(constants.TORCH_COMMAND)
        else:
            runtime.trainer.set_command(constants.DEFAULT_COMMAND)

    trainer_crd = models.TrainerV1alpha1Trainer()

    # Add number of nodes
    if trainer.num_nodes:
        trainer_crd.num_nodes = trainer.num_nodes

    # Add resources per node
    if trainer.resources_per_node:
        trainer_crd.resources_per_node = utils.get_resources_per_node(trainer.resources_per_node)

    # Add environment variables
    if trainer.env:
        trainer_crd.env = [
            models.IoK8sApiCoreV1EnvVar(name=key, value=value) for key, value in trainer.env.items()
        ]

    # Generate function code
    func_code = inspect.getsource(trainer.func)
    func_code = textwrap.dedent(func_code)

    # Generate function call (use **kwargs unpacking like utils.get_command_using_train_func)
    if trainer.func_args is None:
        func_call = f"{trainer.func.__name__}()"
    else:
        # Always unpack kwargs for training function calls
        func_call = f"{trainer.func.__name__}(**{trainer.func_args})"

    func_code = f"{func_code}\n{func_call}\n"

    # Wrap with progression tracking instrumentation if enabled
    if trainer.enable_progression_tracking:
        wrapper_code = get_transformers_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
        )
        func_code = wrapper_code.replace("{{user_func_import_and_call}}", func_code)

    # Inject checkpoint code if enabled
    checkpoint_code = _build_checkpoint_code(trainer)
    if checkpoint_code:
        func_code = f"{checkpoint_code}\n\n{func_code}"

    # Build the command directly with the wrapped function code
    func_file = os.path.basename(inspect.getfile(trainer.func))

    # Install Python packages if required
    install_packages = ""
    if trainer.packages_to_install:
        install_packages = utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
        )

    # Build the trainer command with wrapped function code
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_packages:
                exec_script = install_packages + exec_script
            command.append(exec_script)
        else:
            command.append(c)

    trainer_crd.command = command

    return trainer_crd


def _build_checkpoint_code(trainer: TransformersTrainer) -> str:
    """Generate checkpoint injection code for the trainer."""

    # Only inject if JIT or periodic checkpoint is enabled
    if not trainer.enable_jit_checkpoint and not trainer.periodic_checkpoint_config:
        return ""

    # Create default periodic config if JIT is enabled but no config provided
    periodic_config = trainer.periodic_checkpoint_config
    if trainer.enable_jit_checkpoint and periodic_config is None:
        periodic_config = PeriodicCheckpointConfig()

    # Convert PeriodicCheckpointConfig to dict for injection
    periodic_config_dict = None
    if periodic_config:
        periodic_config_dict = {
            "save_strategy": periodic_config.save_strategy,
            "save_steps": periodic_config.save_steps,
            "save_total_limit": periodic_config.save_total_limit,
        }

    # Parse output_dir URI to get resolved path for checkpoint code
    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    resolved_output_dir, _ = parse_output_dir_uri(trainer.output_dir)

    # Extract S3 URI if output_dir uses S3 storage
    storage_uri = None
    if trainer.output_dir and trainer.output_dir.startswith(S3_URI_SCHEME):
        storage_uri = trainer.output_dir

    # Generate checkpoint injection code
    return get_jit_checkpoint_injection_code(
        output_dir=resolved_output_dir,
        storage_uri=storage_uri,
        periodic_checkpoint_config=periodic_config_dict,
        enable_jit_checkpoint=trainer.enable_jit_checkpoint,
    )


def get_jit_checkpoint_injection_code(
    output_dir: Optional[str] = None,
    storage_uri: Optional[str] = None,
    periodic_checkpoint_config: Optional[dict] = None,
    enable_jit_checkpoint: bool = False,
) -> str:
    """Generate the complete JIT checkpoint code to inject into training scripts."""
    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    # Build checkpoint config dict
    config_dict = {"enable_jit": enable_jit_checkpoint}

    if output_dir:
        config_dict["output_dir"] = output_dir

    if storage_uri:
        config_dict["storage_uri"] = storage_uri

    if periodic_checkpoint_config:
        if "save_strategy" in periodic_checkpoint_config:
            config_dict["save_strategy"] = periodic_checkpoint_config["save_strategy"]
        if "save_steps" in periodic_checkpoint_config:
            config_dict["save_steps"] = periodic_checkpoint_config["save_steps"]
        if "save_total_limit" in periodic_checkpoint_config:
            config_dict["save_total_limit"] = periodic_checkpoint_config["save_total_limit"]

    # Extract the entire function source
    checkpoint_instrumentation_code = inspect.getsource(_create_checkpoint_instrumentation)
    checkpoint_instrumentation_code = textwrap.dedent(checkpoint_instrumentation_code)

    # Remove the import that won't be available in training pods (we'll define it globally instead)
    checkpoint_instrumentation_code = checkpoint_instrumentation_code.replace(
        "from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER",
        "# CHECKPOINT_INCOMPLETE_MARKER defined globally above",
    )

    # Serialize config dict as Python code
    import pprint

    config_dict_str = pprint.pformat(config_dict, indent=4, width=100, sort_dicts=False)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - Checkpoint Instrumentation
# Generated by kubeflow.trainer.rhai.transformers
# =============================================================================

print("[Kubeflow] Initializing checkpoint instrumentation", flush=True)

# Constants (inline to avoid import dependencies in training pods)
CHECKPOINT_INCOMPLETE_MARKER = {repr(CHECKPOINT_INCOMPLETE_MARKER)}

# Instrumentation function definition
{checkpoint_instrumentation_code}

# Initialize and apply instrumentation
checkpoint_config = {config_dict_str}
_, _, apply_checkpointing = _create_checkpoint_instrumentation(checkpoint_config)
apply_checkpointing()
print("[Kubeflow] Checkpoint instrumentation enabled", flush=True)
"""

    return wrapper
