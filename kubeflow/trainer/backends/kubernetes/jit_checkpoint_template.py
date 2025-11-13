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

This module generates the code string to inject into training scripts by
extracting real Python code from jit_checkpoint_code.py using inspect.getsource().
This approach allows the checkpoint code to be navigatable, type-checked, and
maintainable as regular Python code.
"""

import inspect
import textwrap


def get_jit_checkpoint_injection_code() -> str:
    """Generate the complete JIT checkpoint code to inject into training scripts.

    This function extracts real Python code from jit_checkpoint_code.py using
    inspect.getsource(), allowing the checkpoint code to be maintained as
    navigatable, type-checked Python instead of a string literal.

    Returns:
        str: Complete Python code for JIT checkpointing functionality.
    """
    # Import the actual Python code module
    from kubeflow.trainer.backends.kubernetes import jit_checkpoint_code

    # Extract source code for classes and functions
    checkpoint_manager_src = inspect.getsource(jit_checkpoint_code.CheckpointManager)
    callback_src = inspect.getsource(jit_checkpoint_code.JITCheckpointCallback)
    monkey_patch_src = inspect.getsource(jit_checkpoint_code.setup_jit_checkpoint_monkey_patch)

    # Build the complete injection code
    header = textwrap.dedent(
        """
        # ============================================================================
        # JIT Checkpoint Code (Auto-injected by Kubeflow SDK)
        # ============================================================================

        import os
        import signal
        import threading
        from typing import Optional
        from transformers import Trainer as _TransformersTrainer
        from transformers import TrainerCallback
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        from transformers.utils import logging as transformers_logging

        _jit_logger = transformers_logging.get_logger(__name__)

        """
    ).strip()

    # Invoke monkey-patch at module level
    monkey_patch_invocation = textwrap.dedent(
        """

        # ============================================================================
        # Setup Monkey-Patch
        # ============================================================================

        try:
            setup_jit_checkpoint_monkey_patch()
        except ImportError as e:
            _jit_logger.warning(f"Could not import Transformers Trainer for monkey-patching: {e}")
            print(f"Warning: Transformers not available, JIT checkpoint will not work: {e}")
        except Exception as e:
            _jit_logger.error(f"Failed to monkey-patch Trainer.__init__: {e}")
            print(f"Error: Failed to enable JIT checkpoint auto-instrumentation: {e}")

        # ============================================================================
        # End of JIT Checkpoint Code
        # ============================================================================
        """
    ).strip()

    # Combine all parts
    complete_code = (
        f"{header}\n\n\n"
        f"{checkpoint_manager_src}\n\n\n"
        f"{callback_src}\n\n\n"
        f"{monkey_patch_src}\n\n\n"
        f"{monkey_patch_invocation}\n"
    )

    return complete_code