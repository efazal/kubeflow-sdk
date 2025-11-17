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
from typing import Optional


def get_jit_checkpoint_injection_code(
    output_dir: Optional[str] = None,
    periodic_checkpoint_config: Optional[dict] = None,
    enable_jit_checkpoint: bool = False,
) -> str:
    """Generate the complete JIT checkpoint code to inject into training scripts.

    This function extracts real Python code from jit_checkpoint_code.py using
    inspect.getsource(), allowing the checkpoint code to be maintained as
    navigatable, type-checked Python instead of a string literal.

    Args:
        output_dir: Output directory for checkpoints
        periodic_checkpoint_config: Periodic checkpoint configuration dict
            with keys: save_strategy, save_steps, save_total_limit
        enable_jit_checkpoint: Whether to inject JIT checkpoint classes and callback

    Returns:
        str: Complete Python code for JIT checkpointing functionality.
    """
    # Import the actual Python code module
    from kubeflow.trainer.backends.kubernetes import jit_checkpoint_code

    # Extract monkey-patch source (always needed)
    monkey_patch_src = inspect.getsource(jit_checkpoint_code.setup_jit_checkpoint_monkey_patch)

    # Only extract JIT checkpoint classes if JIT is enabled
    checkpoint_manager_src = ""
    callback_src = ""
    if enable_jit_checkpoint:
        checkpoint_manager_src = inspect.getsource(jit_checkpoint_code.CheckpointManager)
        # callback_src = inspect.getsource(jit_checkpoint_code.JITCheckpointCallback)

    # Build checkpoint config dict as Python code
    config_lines = []
    config_lines.append("# ============================================================================")
    config_lines.append("# Kubeflow Checkpoint Configuration (Auto-injected)")
    config_lines.append("# ============================================================================")
    config_lines.append("_KUBEFLOW_CHECKPOINT_CONFIG = {")

    # Add enable_jit flag
    config_lines.append(f'    "enable_jit": {enable_jit_checkpoint},')

    if output_dir:
        config_lines.append(f'    "output_dir": {repr(output_dir)},')

    if periodic_checkpoint_config:
        if "save_strategy" in periodic_checkpoint_config:
            config_lines.append(f'    "save_strategy": {repr(periodic_checkpoint_config["save_strategy"])},')
        if "save_steps" in periodic_checkpoint_config:
            config_lines.append(f'    "save_steps": {periodic_checkpoint_config["save_steps"]},')
        if "save_total_limit" in periodic_checkpoint_config:
            config_lines.append(f'    "save_total_limit": {periodic_checkpoint_config["save_total_limit"]},')

    config_lines.append("}")
    config_code = "\n".join(config_lines)

    # Invoke monkey-patch at module level
    monkey_patch_invocation = textwrap.dedent(
        """

        # ============================================================================
        # Setup Monkey-Patch
        # ============================================================================

        try:
            setup_jit_checkpoint_monkey_patch()
        except ImportError as e:
            print(f"[Kubeflow] Warning: Transformers not available, JIT checkpoint will not work: {e}")
        except Exception as e:
            print(f"[Kubeflow] Error: Failed to enable JIT checkpoint auto-instrumentation: {e}")

        # ============================================================================
        # End of JIT Checkpoint Code
        # ============================================================================
        """
    ).strip()

    # Combine all parts: config dict, (optionally JIT classes), monkey-patch, invocation
    parts = [config_code]

    # Only include JIT classes if they were extracted
    if enable_jit_checkpoint:
        parts.append(checkpoint_manager_src)
        # parts.append(callback_src)

    # Always include monkey-patch and invocation
    parts.append(monkey_patch_src)
    parts.append(monkey_patch_invocation)

    # Join with triple newlines for readability
    complete_code = "\n\n\n".join(parts) + "\n"

    return complete_code