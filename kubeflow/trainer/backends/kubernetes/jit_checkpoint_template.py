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

"""JIT Checkpoint code injection templates for Kubeflow Trainer SDK.

This module provides templates for injecting JIT (Just-In-Time) checkpoint
functionality into CustomTrainer training scripts. When enabled, the checkpoint
code is automatically injected before the user's training function.
"""


def get_jit_checkpoint_injection_code() -> str:
    """Generate the complete JIT checkpoint code to inject into training scripts.

    This function returns a string containing the CheckpointManager and
    JITCheckpointCallback classes that handle SIGTERM-based checkpointing
    for HuggingFace Transformers Trainer.

    Returns:
        str: Complete Python code for JIT checkpointing functionality.
    """
    return '''
# ============================================================================
# JIT Checkpoint Code (Auto-injected by Kubeflow Trainer SDK)
# ============================================================================

import os
import signal
import threading
from typing import Optional
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


# ============================================================================
# Auto-inject JIT Checkpoint Callback via Trainer Monkey-Patching
# ============================================================================

try:
    from transformers import Trainer as _TransformersTrainer

    # Create singleton callback instance
    _jit_checkpoint_callback = JITCheckpointCallback()

    # Store the original __init__ method
    _original_trainer_init = _TransformersTrainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
        # Get existing callbacks
        callbacks = kwargs.get('callbacks', [])
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
        kwargs['callbacks'] = callbacks

        # Call original __init__
        _original_trainer_init(self, *args, **kwargs)

        # CRITICAL: Store trainer reference in the callback for later use
        _jit_checkpoint_callback._trainer_ref = self
        _jit_logger.debug("Stored Trainer reference in JIT checkpoint callback")

    # Apply the monkey-patch
    _TransformersTrainer.__init__ = _patched_trainer_init
    _jit_logger.info("Successfully monkey-patched Trainer.__init__ for JIT checkpointing")
    print("Kubeflow JIT Checkpoint: Trainer auto-instrumentation enabled")

except ImportError as e:
    _jit_logger.warning(f"Could not import Transformers Trainer for monkey-patching: {e}")
    print(f"Warning: Transformers not available, JIT checkpoint will not work: {e}")
except Exception as e:
    _jit_logger.error(f"Failed to monkey-patch Trainer.__init__: {e}")
    print(f"Error: Failed to enable JIT checkpoint auto-instrumentation: {e}")

# ============================================================================
# End of JIT Checkpoint Code
# ============================================================================
'''