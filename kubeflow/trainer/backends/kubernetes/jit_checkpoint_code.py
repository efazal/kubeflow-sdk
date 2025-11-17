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
user training scripts when JIT checkpointing is enabled. Each class/function
is self-contained with its own imports for easy injection via inspect.getsource().
"""


class CheckpointManager:
    """Manages just-in-time checkpointing on SIGTERM signal."""

    import os
    import signal
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    from transformers import TrainerCallback

    def __init__(self, trainer):
        self.trainer = trainer
        self.checkpoint_requested = False
        self._original_sigterm_handler = None

    def setup_signal_handler(self):
        """Register SIGTERM signal handler for JIT checkpointing."""
        self._original_sigterm_handler = self.signal.signal(self.signal.SIGTERM, self._sigterm_handler)
        print("[Kubeflow] JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM by requesting immediate checkpoint."""
        if not self.checkpoint_requested:
            print("[Kubeflow] SIGTERM received, requesting JIT checkpoint")
            self.checkpoint_requested = True

    def execute_jit_checkpoint(self):
        """Execute the actual checkpoint save operation."""
        try:
            self.checkpoint_requested = False
            current_step = self.trainer.state.global_step

            print(f"[Kubeflow] Starting JIT checkpoint at step {current_step}")

            output_dir = self.trainer._get_output_dir(trial=None)
            checkpoint_path = self.os.path.join(output_dir, f"{self.PREFIX_CHECKPOINT_DIR}-{current_step}")
            self.os.makedirs(checkpoint_path, exist_ok=True)

            # Create sentinel file to mark incomplete checkpoint
            sentinel_file = self.os.path.join(checkpoint_path, "checkpoint-is-incomplete.txt")
            with open(sentinel_file, "w") as f:
                f.write(f"Checkpoint started at step {current_step}")

            # Save checkpoint
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

            # Remove sentinel on success
            if self.os.path.exists(sentinel_file):
                self.os.remove(sentinel_file)

            print("[Kubeflow] JIT checkpoint completed successfully")

        except Exception as e:
            print(f"[Kubeflow] Failed to save JIT checkpoint: {e}")
            raise

    def should_checkpoint_now(self):
        """Check if a checkpoint has been requested."""
        return self.checkpoint_requested


    class JITCheckpointCallback(TrainerCallback):
        """Transformers callback that integrates JIT checkpointing with trainer lifecycle."""

        def __init__(self):
            self.jit_manager = None
            self._trainer_ref = None

        def on_train_begin(self, args, state, control, **kwargs):
            """Initialize checkpoint manager when training begins."""
            if self._trainer_ref is not None and self.jit_manager is None:
                self.jit_manager = CheckpointManager(trainer=self._trainer_ref)
                self.jit_manager.setup_signal_handler()
                print("[Kubeflow] JIT checkpointing enabled")
            elif self._trainer_ref is None:
                print("[Kubeflow] Warning: Trainer reference not set for JIT checkpoint callback")

        def _trigger_checkpoint_if_requested(self, control):
            """Check and trigger checkpoint if SIGTERM was received."""
            if self.jit_manager and self.jit_manager.should_checkpoint_now():
                control.should_save = False
                control.should_training_stop = True
                self.jit_manager.execute_jit_checkpoint()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self._trigger_checkpoint_if_requested(control)

        def on_step_begin(self, args, state, control, **kwargs):
            self._trigger_checkpoint_if_requested(control)

        def on_step_end(self, args, state, control, **kwargs):
            self._trigger_checkpoint_if_requested(control)

        def on_epoch_end(self, args, state, control, **kwargs):
            self._trigger_checkpoint_if_requested(control)


def setup_jit_checkpoint_monkey_patch():
    """Setup monkey-patch for Trainer to auto-inject JIT checkpoint callback."""
    import os
    import re
    import shutil
    from transformers import Trainer as _TransformersTrainer
    from transformers import TrainingArguments

    # Singleton callback instance
    _jit_checkpoint_callback = CheckpointManager.JITCheckpointCallback()

    # Store original __init__ methods
    _original_trainer_init = _TransformersTrainer.__init__
    _original_training_args_init = TrainingArguments.__init__

    def _find_latest_checkpoint(output_dir):
        """Find the latest checkpoint, deleting incomplete ones."""
        if not output_dir or not os.path.exists(output_dir):
            return None

        checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
        checkpoints = []

        for name in os.listdir(output_dir):
            match = checkpoint_pattern.match(name)
            if not match or not os.path.isdir(os.path.join(output_dir, name)):
                continue

            checkpoint_path = os.path.join(output_dir, name)
            incomplete_marker = os.path.join(checkpoint_path, "checkpoint-is-incomplete.txt")

            # Delete incomplete checkpoints
            if os.path.exists(incomplete_marker):
                print(f"[Kubeflow] Deleting incomplete checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
                continue

            checkpoints.append((int(match.group(1)), checkpoint_path))

        if checkpoints:
            checkpoints.sort(reverse=True)
            latest = checkpoints[0][1]
            print(f"[Kubeflow] Found latest checkpoint: {latest}")
            return latest

        return None

    def _patched_training_args_init(self, *args, **kwargs):
        """Patched TrainingArguments.__init__ that applies checkpoint defaults."""
        config = globals().get("_KUBEFLOW_CHECKPOINT_CONFIG", {})

        # Apply Kubeflow config defaults if not explicitly set by user
        for key in ["output_dir", "save_strategy", "save_steps", "save_total_limit"]:
            if key in config and key not in kwargs:
                kwargs[key] = config[key]
                print(f"[Kubeflow] Applied {key}: {config[key]}")

        _original_training_args_init(self, *args, **kwargs)

    def _patched_trainer_init(self, *args, **kwargs):
        """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
        config = globals().get("_KUBEFLOW_CHECKPOINT_CONFIG", {})
        enable_jit = config.get("enable_jit", False)

        # Extract training_args from args or kwargs
        training_args = None
        if args and hasattr(args[0], "output_dir"):
            training_args = args[0]
        elif "args" in kwargs and hasattr(kwargs["args"], "output_dir"):
            training_args = kwargs["args"]

        # Override training_args attributes for TrlParser case (bypasses __init__)
        if training_args and config:
            # Override output_dir if it's a default value
            if "output_dir" in config and training_args.output_dir in ["trainer_output", "tmp_trainer", "output"]:
                training_args.output_dir = config["output_dir"]
                print(f"[Kubeflow] Overrode output_dir: {config['output_dir']}")

            # Override save_strategy if it's the default "epoch"
            save_strategy_changed = False
            if "save_strategy" in config and str(training_args.save_strategy) == "epoch":
                training_args.save_strategy = config["save_strategy"]
                save_strategy_changed = True
                print(f"[Kubeflow] Overrode save_strategy: {config['save_strategy']}")

            # Override save_steps if strategy changed or value is default
            if "save_steps" in config:
                if save_strategy_changed or training_args.save_steps in (None, 500):
                    training_args.save_steps = config["save_steps"]
                    print(f"[Kubeflow] Overrode save_steps: {config['save_steps']}")

            # Override save_total_limit if strategy changed or value is None
            if "save_total_limit" in config:
                if save_strategy_changed or training_args.save_total_limit is None:
                    training_args.save_total_limit = config["save_total_limit"]
                    print(f"[Kubeflow] Overrode save_total_limit: {config['save_total_limit']}")

        # Inject JIT callback if enabled
        if enable_jit:
            callbacks = kwargs.get("callbacks") or []
            if not isinstance(callbacks, list):
                callbacks = list(callbacks)
            if not any(isinstance(cb, JITCheckpointCallback) for cb in callbacks):
                callbacks.append(_jit_checkpoint_callback)
                print("[Kubeflow] Auto-injected JIT checkpoint callback")
            kwargs["callbacks"] = callbacks

        # Call original __init__
        _original_trainer_init(self, *args, **kwargs)

        # Store trainer reference in callback
        if enable_jit:
            _jit_checkpoint_callback._trainer_ref = self

        # Monkey-patch train() method for auto-resume
        _original_train = self.train
        def _patched_train(resume_from_checkpoint=None, **train_kwargs):
            """Patched train() that auto-resumes from latest checkpoint if available."""
            # Only auto-resume if user didn't explicitly set it
            if resume_from_checkpoint is None and training_args:
                latest_checkpoint = _find_latest_checkpoint(training_args.output_dir)
                if latest_checkpoint:
                    resume_from_checkpoint = latest_checkpoint
                    print(f"[Kubeflow] Auto-resuming from: {latest_checkpoint}")
            return _original_train(resume_from_checkpoint=resume_from_checkpoint, **train_kwargs)

        self.train = _patched_train

    # Apply monkey-patches
    TrainingArguments.__init__ = _patched_training_args_init
    _TransformersTrainer.__init__ = _patched_trainer_init
    print("[Kubeflow] Trainer auto-instrumentation enabled")