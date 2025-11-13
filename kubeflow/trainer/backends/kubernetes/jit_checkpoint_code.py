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

"""JIT Checkpoint classes for injection into training scripts.

This module contains the actual Python code that will be injected into
user training scripts when JIT checkpointing is enabled. The code here
is real Python that can be navigated, type-checked, and tested.
"""

import os
import signal
import threading
from typing import Optional

from transformers import Trainer as _TransformersTrainer
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging as transformers_logging

_jit_logger = transformers_logging.get_logger(__name__)


class CheckpointManager:
    """Manages just-in-time checkpointing on SIGTERM signal.

    This class sets up a SIGTERM handler that triggers a checkpoint
    after a configurable wait period, allowing graceful checkpoint
    completion before pod termination.
    """

    def __init__(self, trainer, kill_wait: int = 3):
        """Initialize the checkpoint manager.

        Args:
            trainer: HuggingFace Transformers Trainer instance.
            kill_wait: Seconds to wait after SIGTERM before checkpointing (default: 3).
        """
        self.trainer = trainer
        self.checkpoint_requested = False
        self._original_sigterm_handler = None
        self.kill_wait = kill_wait

    def setup_signal_handler(self):
        """Register SIGTERM signal handler for JIT checkpointing."""
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        _jit_logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM by scheduling a checkpoint after kill_wait period."""
        if self.checkpoint_requested:
            return

        _jit_logger.info(f"SIGTERM received, will request JIT checkpoint after {self.kill_wait}s")
        threading.Timer(self.kill_wait, self._toggle_checkpoint_flag).start()

    def _toggle_checkpoint_flag(self):
        """Set the checkpoint flag after the kill wait period."""
        _jit_logger.info("Kill wait period elapsed, requesting checkpoint")
        self.checkpoint_requested = True

    def execute_jit_checkpoint(self):
        """Execute the actual checkpoint save operation."""
        try:
            # Set checkpoint flag to False to avoid multiple checkpoints
            self.checkpoint_requested = False

            _jit_logger.info("Starting JIT checkpointing...")
            current_step = self.trainer.state.global_step
            _jit_logger.info(f"Saving JIT checkpoint at step {current_step}")

            output_dir = self.trainer._get_output_dir(trial=None)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{current_step}"
            checkpoint_path = os.path.join(output_dir, checkpoint_folder)

            # Create checkpoint directory
            os.makedirs(checkpoint_path, exist_ok=True)

            # Create a sentinel file to indicate checkpointing is in progress
            sentinel_file = os.path.join(output_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")
            with open(sentinel_file, "w") as f:
                f.write(f"Checkpoint started at step {current_step} and in progress...")
            _jit_logger.info(f"Created checkpoint progress sentinel marker file: {sentinel_file}")

            # Invoke the trainer's checkpoint method directly
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

            # Remove sentinel file upon successful checkpointing
            if os.path.exists(sentinel_file):
                os.remove(sentinel_file)
                _jit_logger.info("Sentinel marker file removed")

            _jit_logger.info("JIT checkpoint completed successfully")
            print("JIT checkpoint completed successfully")

        except Exception as e:
            _jit_logger.error(f"Failed to save JIT checkpoint: {e}")
            raise

    def should_checkpoint_now(self) -> bool:
        """Check if a checkpoint has been requested.

        Returns:
            bool: True if checkpoint should be executed now.
        """
        return self.checkpoint_requested


class JITCheckpointCallback(TrainerCallback):
    """Transformers callback that integrates JIT checkpointing with trainer lifecycle.

    This callback monitors the training process and triggers checkpoints
    when SIGTERM is received, ensuring graceful pod termination.
    """

    def __init__(self):
        """Initialize the JIT checkpoint callback."""
        self.jit_manager: Optional[CheckpointManager] = None
        self._trainer_ref = None  # Will be set by monkey-patch

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize checkpoint manager when training begins."""
        if self._trainer_ref is not None and self.jit_manager is None:
            self.jit_manager = CheckpointManager(trainer=self._trainer_ref)
            self.jit_manager.setup_signal_handler()
            _jit_logger.info("JIT checkpointing enabled")
            print("JIT checkpointing enabled")
        else:
            if self._trainer_ref is None:
                _jit_logger.warning("Trainer reference not set for JIT checkpoint callback")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Check for checkpoint request before optimizer step."""
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_step_begin(self, args, state, control, **kwargs):
        """Check for checkpoint request at step begin."""
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_step_end(self, args, state, control, **kwargs):
        """Check for checkpoint request at step end."""
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_save = False
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Check for checkpoint request at epoch end."""
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_save = False
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()


def setup_jit_checkpoint_monkey_patch():
    """Setup monkey-patch for Trainer to auto-inject JIT checkpoint callback."""
    from transformers import TrainingArguments

    # Create singleton callback instance
    _jit_checkpoint_callback = JITCheckpointCallback()

    # Store the original __init__ methods
    _original_trainer_init = _TransformersTrainer.__init__
    _original_training_args_init = TrainingArguments.__init__

    def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
        """Find the latest checkpoint in the output directory.

        Args:
            output_dir: Directory to search for checkpoints.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found.
        """
        if not output_dir or not os.path.exists(output_dir):
            return None

        import re

        checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
        checkpoints = []

        for name in os.listdir(output_dir):
            match = checkpoint_pattern.match(name)
            if match:
                checkpoint_path = os.path.join(output_dir, name)
                # Verify it's a valid checkpoint (has config or model files)
                if os.path.isdir(checkpoint_path):
                    # Check for incomplete checkpoint marker
                    incomplete_marker = os.path.join(checkpoint_path, "checkpoint-is-incomplete.txt")
                    if os.path.exists(incomplete_marker):
                        _jit_logger.warning(f"Skipping incomplete checkpoint: {checkpoint_path}")
                        continue

                    # Valid checkpoint
                    step = int(match.group(1))
                    checkpoints.append((step, checkpoint_path))

        if not checkpoints:
            return None

        # Return the checkpoint with the highest step number
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoints[0][1]
        _jit_logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def _patched_training_args_init(self, *args, **kwargs):
        """Patched TrainingArguments.__init__ that applies checkpoint defaults.

        Only applies defaults for arguments that the user hasn't explicitly set.
        """
        # Get Kubeflow-injected configs from environment variables
        import json

        kubeflow_output_dir = os.environ.get("KUBEFLOW_OUTPUT_DIR")
        kubeflow_checkpoint_config = os.environ.get("KUBEFLOW_CHECKPOINT_CONFIG")

        # Parse checkpoint config
        checkpoint_config = {}
        if kubeflow_checkpoint_config:
            try:
                checkpoint_config = json.loads(kubeflow_checkpoint_config)
            except json.JSONDecodeError:
                _jit_logger.warning("Failed to parse KUBEFLOW_CHECKPOINT_CONFIG")

        # Apply output_dir if not set by user
        if kubeflow_output_dir and "output_dir" not in kwargs:
            kwargs["output_dir"] = kubeflow_output_dir
            _jit_logger.info(f"Applied Kubeflow output_dir: {kubeflow_output_dir}")

        # Apply periodic checkpoint config defaults (only if not explicitly set by user)
        if checkpoint_config:
            if "save_strategy" in checkpoint_config and "save_strategy" not in kwargs:
                kwargs["save_strategy"] = checkpoint_config["save_strategy"]
                _jit_logger.info(f"Applied save_strategy: {checkpoint_config['save_strategy']}")

            if "save_steps" in checkpoint_config and "save_steps" not in kwargs:
                kwargs["save_steps"] = checkpoint_config["save_steps"]
                _jit_logger.info(f"Applied save_steps: {checkpoint_config['save_steps']}")

            if "save_total_limit" in checkpoint_config and "save_total_limit" not in kwargs:
                kwargs["save_total_limit"] = checkpoint_config["save_total_limit"]
                _jit_logger.info(f"Applied save_total_limit: {checkpoint_config['save_total_limit']}")

            if "load_best_model_at_end" in checkpoint_config and "load_best_model_at_end" not in kwargs:
                kwargs["load_best_model_at_end"] = checkpoint_config["load_best_model_at_end"]

        # Call original __init__
        _original_training_args_init(self, *args, **kwargs)

    def _patched_trainer_init(self, *args, **kwargs):
        """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
        # Get existing callbacks
        callbacks = kwargs.get("callbacks", [])
        if callbacks is None:
            callbacks = []

        # Ensure it's a list
        if not isinstance(callbacks, list):
            callbacks = list(callbacks)

        # Add JIT checkpoint callback if not already present
        has_jit_callback = any(isinstance(cb, JITCheckpointCallback) for cb in callbacks)
        if not has_jit_callback:
            callbacks.append(_jit_checkpoint_callback)
            _jit_logger.info("Auto-injected JIT checkpoint callback into Trainer")

        # Update kwargs with modified callbacks
        kwargs["callbacks"] = callbacks

        # Smart resume from checkpoint (only if user didn't explicitly set it)
        if "resume_from_checkpoint" not in kwargs:
            # Get args - could be first positional arg or in kwargs
            training_args = None
            if args and hasattr(args[0], "output_dir"):
                training_args = args[0]
            elif "args" in kwargs and hasattr(kwargs["args"], "output_dir"):
                training_args = kwargs["args"]

            if training_args and hasattr(training_args, "output_dir"):
                latest_checkpoint = _find_latest_checkpoint(training_args.output_dir)
                if latest_checkpoint:
                    kwargs["resume_from_checkpoint"] = latest_checkpoint
                    _jit_logger.info(f"Auto-resuming from checkpoint: {latest_checkpoint}")
                    print(f"Kubeflow: Auto-resuming training from {latest_checkpoint}")

        # Call original __init__
        _original_trainer_init(self, *args, **kwargs)

        _jit_checkpoint_callback._trainer_ref = self
        _jit_logger.debug("Stored Trainer reference in JIT checkpoint callback")

    # Apply the monkey-patches
    TrainingArguments.__init__ = _patched_training_args_init
    _TransformersTrainer.__init__ = _patched_trainer_init
    _jit_logger.info("Successfully monkey-patched Trainer and TrainingArguments for checkpointing")
    print("Kubeflow JIT Checkpoint: Trainer auto-instrumentation enabled")