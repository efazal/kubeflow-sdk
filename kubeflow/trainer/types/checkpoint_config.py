# Copyright 2025 The Kubeflow Authors.
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

"""Checkpoint configuration types for Kubeflow Trainer SDK."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SaveStrategy(Enum):
    """Strategy for saving checkpoints."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


@dataclass
class PeriodicCheckpointConfig:
    """Configuration for periodic checkpoint saving.

    This config controls how often checkpoints are saved during training.
    When enabled with JIT checkpoint, both periodic and JIT checkpoints will work together:
    - Periodic checkpoints save at regular intervals (steps/epochs)
    - JIT checkpoints save on SIGTERM signal (pod preemption)

    Args:
        save_strategy (`SaveStrategy` or `str`): The checkpoint save strategy to use.
            Can be "no", "steps", or "epoch". Defaults to "steps".
        save_steps (`Optional[int]`): Save checkpoint every X training steps.
            Only used when save_strategy="steps". Defaults to 500.
        save_total_limit (`Optional[int]`): Maximum number of checkpoints to keep.
            Older checkpoints are deleted. If None, all checkpoints are kept.
            Defaults to 3 to save storage space.
        load_best_model_at_end (`bool`): Whether to load the best model at the end of training.
            Requires evaluation strategy to be enabled. Defaults to False.

    Example:
        ```python
        from kubeflow.trainer import CustomTrainer, PeriodicCheckpointConfig

        checkpoint_config = PeriodicCheckpointConfig(
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=5
        )

        trainer = CustomTrainer(
            func=train_function,
            periodic_checkpoint_config=checkpoint_config
        )
        ```
    """

    save_strategy: SaveStrategy | str = SaveStrategy.EPOCH
    save_steps: Optional[int] = 500
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = False

    def __post_init__(self):
        """Validate configuration."""
        # Convert string to enum if needed
        if isinstance(self.save_strategy, str):
            try:
                self.save_strategy = SaveStrategy(self.save_strategy.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid save_strategy: {self.save_strategy}. "
                    f"Must be one of: {[s.value for s in SaveStrategy]}"
                )

        # Validate save_steps
        if self.save_strategy == SaveStrategy.STEPS:
            if self.save_steps is None or self.save_steps <= 0:
                raise ValueError("save_steps must be a positive integer when save_strategy='steps'")

        # Validate save_total_limit
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be a positive integer or None")


@dataclass
class S3StorageConfig:
    """Configuration for S3-compatible storage for checkpoints.

    Args:
        endpoint (`str`): S3 endpoint URL (e.g., "https://s3.amazonaws.com").
        bucket (`str`): S3 bucket name where checkpoints will be saved.
        access_key_id (`Optional[str]`): AWS access key ID. If None, uses environment variables
            or IAM role.
        secret_access_key (`Optional[str]`): AWS secret access key. If None, uses environment
            variables or IAM role.
        region (`Optional[str]`): AWS region (e.g., "us-east-1"). Defaults to "us-east-1".
        path_prefix (`Optional[str]`): Prefix path within the bucket (e.g., "checkpoints/model1").

    Example:
        ```python
        from kubeflow.trainer import CustomTrainer, S3StorageConfig

        storage = S3StorageConfig(
            endpoint="https://s3.amazonaws.com",
            bucket="my-training-checkpoints",
            region="us-west-2",
            path_prefix="llama3-fine-tune"
        )

        trainer = CustomTrainer(
            func=train_function,
            output_dir="s3://my-training-checkpoints/llama3-fine-tune",
            storage_config=storage
        )
        ```
    """

    endpoint: str
    bucket: str
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: Optional[str] = "us-east-1"
    path_prefix: Optional[str] = None

    def __post_init__(self):
        """Validate S3 configuration."""
        if not self.endpoint:
            raise ValueError("S3 endpoint must be specified")
        if not self.bucket:
            raise ValueError("S3 bucket must be specified")


@dataclass
class PVCStorageConfig:
    """Configuration for Kubernetes PVC storage for checkpoints.

    Args:
        claim_name (`str`): Name of the PersistentVolumeClaim to use.
        mount_path (`str`): Path where the PVC will be mounted in the pod.
            Defaults to "/mnt/checkpoints".

    Example:
        ```python
        from kubeflow.trainer import CustomTrainer, PVCStorageConfig

        storage = PVCStorageConfig(
            claim_name="training-checkpoints-pvc",
            mount_path="/mnt/training"
        )

        trainer = CustomTrainer(
            func=train_function,
            output_dir="/mnt/training/llama3-fine-tune",
            storage_config=storage
        )
        ```
    """

    claim_name: str
    mount_path: str = "/mnt/checkpoints"

    def __post_init__(self):
        """Validate PVC configuration."""
        if not self.claim_name:
            raise ValueError("PVC claim_name must be specified")
        if not self.mount_path:
            raise ValueError("PVC mount_path must be specified")
        if not self.mount_path.startswith("/"):
            raise ValueError(f"PVC mount_path must be an absolute path: {self.mount_path}")