from dataclasses import dataclass, field
from enum import Enum
import inspect
import textwrap

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class SpeculatorMode(Enum):
    """Pipeline mode for speculator training."""

    TRAIN_ONLY = "train_only"
    DATA_ONLY = "data_only"
    OFFLINE = "offline"
    ONLINE = "online"


class SpeculatorType(Enum):
    """Draft model architecture for speculative decoding."""

    EAGLE3 = "eagle3"
    DFLASH = "dflash"
    MTP = "mtp"
    PEAGLE = "peagle"


VALID_DTYPES = ("bfloat16", "float16", "float32")
VALID_LOSS_FNS = ("kl_div", "ce")
VALID_SCHEDULERS = ("linear", "cosine", "none")


@dataclass
class SpeculatorConfig:
    """Advanced configuration for speculator training.

    These parameters have sensible defaults based on speculators upstream
    recommendations. Most users won't need to change them. Pass as
    ``config=SpeculatorConfig(...)`` to SpeculatorTrainer.

    Model Architecture:
        num_layers: Number of draft model decoder layers. Single layer gives
            the best speed/accuracy tradeoff for most models.
        ttt_steps: Test-time training steps. Simulates multi-step speculative
            decoding during training for better acceptance rates.
        norm_before_residual: Apply layer norm before residual connection.
        norm_before_fc: Apply layer norm before fully connected layer.
        embed_requires_grad: Whether the embedding layer is trainable.
        hidden_states_dtype: PyTorch dtype for hidden states tensors.

    Training:
        scheduler_type: Learning rate scheduler (linear, cosine, none).
        loss_fn: Loss function. kl_div preserves probability distribution
            (recommended). ce is cross-entropy with argmax labels.
        noise_std: Gaussian noise std for data augmentation.
        checkpoint_freq: Checkpoint save frequency in epochs (1.0 = every epoch).
        log_freq: Metric logging frequency in training steps.

    Data Generation:
        datagen_concurrency: Parallel requests for hidden state extraction.
            Higher values speed up extraction but use more memory.
        target_layer_ids: Verifier layer IDs to extract hidden states from.
            Auto-computed as [2, n//2, n-3] if not provided. Only change
            this if you used custom layers during vLLM data generation.

    Resume:
        from_pretrained: Path to a checkpoint to resume training from.
        use_off_policy_tokens: Use off-policy (ground truth) tokens during
            training instead of on-policy (draft-generated) tokens.
        ttt_step_loss_decay: Loss decay factor per TTT step. 1.0 means
            equal weight for all steps.
    """

    # Model architecture
    num_layers: int = 1
    ttt_steps: int = 3
    norm_before_residual: bool = True
    norm_before_fc: bool = False
    embed_requires_grad: bool = False
    hidden_states_dtype: str = "bfloat16"

    # Training
    scheduler_type: str = "linear"
    loss_fn: str = "kl_div"
    noise_std: float = 0.05
    checkpoint_freq: float = 1.0
    log_freq: int = 1

    # Data generation
    datagen_concurrency: int = 4
    target_layer_ids: list[int] | None = None

    # Resume and advanced training
    from_pretrained: str | None = None
    use_off_policy_tokens: bool = False
    ttt_step_loss_decay: float = 1.0

    def __post_init__(self):
        if self.hidden_states_dtype not in VALID_DTYPES:
            raise ValueError(
                f"hidden_states_dtype must be one of {VALID_DTYPES}, "
                f"got '{self.hidden_states_dtype}'."
            )
        if self.loss_fn not in VALID_LOSS_FNS:
            raise ValueError(f"loss_fn must be one of {VALID_LOSS_FNS}, got '{self.loss_fn}'.")
        if self.scheduler_type not in VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler_type must be one of {VALID_SCHEDULERS}, got '{self.scheduler_type}'."
            )


@dataclass
class SpeculatorTrainer:
    """Speculator training for draft models via Kubeflow Trainer.

    Trains lightweight draft models (Eagle3, DFlash, MTP, PEagle) for
    speculative decoding inference acceleration. The SDK auto-generates
    the training script and configures the vLLM sidecar.

    Four training modes:
        - train_only: Train from pre-extracted hidden states (model-opt CTR)
        - data_only: Extract hidden states via vLLM sidecar (sidecar CTR)
        - offline: Extract + train with user-managed external vLLM (model-opt CTR)
        - online: Train with on-the-fly hidden state generation (sidecar CTR)

    Args:
        mode: Pipeline mode. Determines which CTR to use and what steps run.
        verifier_model: HuggingFace model ID or local path to the verifier model.
        output_dir: PVC URI for saving all artifacts (pvc://<name>/<path>).
            The SDK auto-mounts the PVC. Required for all modes.
        speculator_type: Draft model architecture to train.
        epochs: Number of training epochs.
        lr: Learning rate.
        total_seq_len: Maximum sequence length for training and preprocessing.
        draft_vocab_size: Draft model vocabulary size. None auto-detects from
            token frequencies (recommended).
        hidden_states_path: PVC URI to pre-generated hidden states.
            Required for train_only mode. Must use the same PVC as output_dir.
        data_path: Local path to the Arrow dataset directory.
            Required for train_only mode.
        dataset_name: Built-in dataset name for preprocessing (sharegpt, ultrachat).
            Used by data_only, offline, and online modes.
        max_samples: Maximum number of dataset samples to use.
        regenerate_responses: Run response regeneration before preprocessing
            in data_only mode. Regenerates responses from the verifier model
            for better training data quality.
        vllm_endpoint: URL of a running vLLM server with extraction config.
            Required for offline mode.
        vllm_gpu_count: Number of GPUs for the vLLM sidecar. Controls
            tensor parallel size for large verifier models.
        vllm_gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0 to 1.0).
            Lower values leave more memory for training in online mode.
        training_gpu_count: Number of GPUs for training. Auto-sets
            resources_per_node if not explicitly provided.
        trust_remote_code: Trust remote code when loading HuggingFace models.
        config: Advanced speculator configuration. Pass SpeculatorConfig(...)
            to tune model architecture, loss function, scheduler, and other
            advanced parameters. None uses sensible defaults.
        packages_to_install: Python packages to install before training.
        pip_index_urls: PyPI index URLs for package installation.
        resources_per_node: Computing resources per node. Overrides
            training_gpu_count if provided.
        env: Additional environment variables for training pods.
    """

    mode: SpeculatorMode
    verifier_model: str
    output_dir: str | None = None
    speculator_type: SpeculatorType = SpeculatorType.EAGLE3

    # Training essentials
    epochs: int = 3
    lr: float = 1e-4
    total_seq_len: int = 2048
    draft_vocab_size: int | None = None

    # Data
    hidden_states_path: str | None = None
    data_path: str | None = None
    dataset_name: str = "sharegpt"
    max_samples: int | None = None
    regenerate_responses: bool = False

    # vLLM
    vllm_endpoint: str | None = None
    vllm_gpu_count: int = 1
    vllm_gpu_memory_utilization: float = 0.9

    # Training GPU
    training_gpu_count: int = 1

    # Advanced config (optional)
    config: SpeculatorConfig | None = None

    # Progression tracking
    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    # Infrastructure
    trust_remote_code: bool = True
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    resources_per_node: dict | None = None
    env: dict[str, str] | None = None

    def __post_init__(self):
        from kubeflow.trainer.rhai.constants import PVC_URI_SCHEME, S3_URI_SCHEME

        if self.verifier_model is None:
            raise ValueError("'verifier_model' is required.")

        if not self.output_dir:
            raise ValueError(
                "'output_dir' is required. Provide a PVC URI (pvc://<pvc-name>/<path>) "
                "for saving training artifacts."
            )

        if self.output_dir.startswith(S3_URI_SCHEME):
            raise ValueError(
                "S3 output_dir is not supported for speculator training. "
                "Use a PVC URI (pvc://<pvc-name>/<path>) instead."
            )
        from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

        self.output_dir = normalize_and_validate_output_dir(self.output_dir)

        if self.hidden_states_path:
            if self.hidden_states_path.startswith(S3_URI_SCHEME):
                raise ValueError(
                    "S3 hidden_states_path is not supported for speculator training. "
                    "Use a PVC URI (pvc://<pvc-name>/<path>) instead."
                )
            self.hidden_states_path = normalize_and_validate_output_dir(self.hidden_states_path)

            if self.output_dir.startswith(PVC_URI_SCHEME) and self.hidden_states_path.startswith(
                PVC_URI_SCHEME
            ):
                output_pvc = self.output_dir[len(PVC_URI_SCHEME) :].split("/")[0]
                hs_pvc = self.hidden_states_path[len(PVC_URI_SCHEME) :].split("/")[0]
                if output_pvc != hs_pvc:
                    raise ValueError(
                        f"output_dir and hidden_states_path must use the same PVC. "
                        f"Got output_dir PVC '{output_pvc}' and hidden_states_path "
                        f"PVC '{hs_pvc}'. Use the same PVC for both."
                    )

        if not isinstance(self.speculator_type, SpeculatorType):
            raise ValueError(
                f"speculator_type must be a SpeculatorType enum value, "
                f"got {self.speculator_type!r}. "
                f"Valid options: {[t.value for t in SpeculatorType]}"
            )

        if not isinstance(self.vllm_gpu_count, int) or self.vllm_gpu_count < 1:
            raise ValueError(
                f"vllm_gpu_count must be a positive integer, got {self.vllm_gpu_count!r}."
            )

        if (
            not isinstance(self.vllm_gpu_memory_utilization, (int, float))
            or self.vllm_gpu_memory_utilization <= 0
            or self.vllm_gpu_memory_utilization > 1.0
        ):
            raise ValueError(
                f"vllm_gpu_memory_utilization must be between 0 and 1.0, "
                f"got {self.vllm_gpu_memory_utilization!r}."
            )

        if not isinstance(self.training_gpu_count, int) or self.training_gpu_count < 1:
            raise ValueError(
                f"training_gpu_count must be a positive integer, got {self.training_gpu_count!r}."
            )

        if self.config is None:
            self.config = SpeculatorConfig()

        if not isinstance(self.metrics_port, int) or not (1024 <= self.metrics_port <= 65535):
            raise ValueError(
                f"metrics_port must be an integer in range 1024-65535, got {self.metrics_port!r}."
            )
        if not isinstance(self.metrics_poll_interval_seconds, int) or not (
            5 <= self.metrics_poll_interval_seconds <= 300
        ):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer in range 5-300, "
                f"got {self.metrics_poll_interval_seconds!r}."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY:
            if not self.hidden_states_path:
                raise ValueError(
                    "'train_only' mode requires 'hidden_states_path' pointing "
                    "to pre-generated hidden states (pvc://<pvc-name>/<path>)."
                )
            if not self.data_path:
                raise ValueError(
                    "'train_only' mode requires 'data_path' pointing to the "
                    "Arrow dataset directory."
                )

        elif self.mode == SpeculatorMode.OFFLINE:
            if not self.vllm_endpoint:
                raise ValueError(
                    "'offline' mode requires 'vllm_endpoint' pointing to a "
                    "running vLLM server with extraction config."
                )


# =============================================================================
# Pod-injected functions for each training mode.
#
# These functions are NOT called directly in the SDK. They are extracted as
# source code via inspect.getsource() and injected into training pod commands.
# This provides syntax highlighting, testability, and type checking while
# avoiding runtime SDK dependencies inside the container.
# =============================================================================


def _create_speculator_progression_server(metrics_port: int) -> tuple:
    """Create an HTTP metrics server for progression tracking.

    Extracted via inspect.getsource() and injected into training pods.
    Returns helper functions for updating and serving metrics.
    """
    from dataclasses import asdict, dataclass, field
    import http.server
    import json
    import threading
    import time
    from typing import Any

    @dataclass
    class ProgressionMetricsState:
        """camelCase fields for Kubernetes API compatibility."""

        progressPercentage: int | None = None  # noqa: N815
        estimatedRemainingSeconds: int | None = None  # noqa: N815
        currentStep: int = 0  # noqa: N815
        totalSteps: int | None = None  # noqa: N815
        currentEpoch: float = 0.0  # noqa: N815
        totalEpochs: int | None = None  # noqa: N815
        currentPhase: str | None = None  # noqa: N815
        trainMetrics: dict[str, Any] = field(default_factory=dict)  # noqa: N815

    _state = ProgressionMetricsState()
    _lock = threading.Lock()

    def _update_metrics(updates: dict) -> None:
        with _lock:
            for key, value in updates.items():
                if hasattr(_state, key):
                    current = getattr(_state, key)
                    if isinstance(value, dict) and isinstance(current, dict):
                        current.update(value)
                    else:
                        setattr(_state, key, value)

    def _get_metrics_json() -> str:
        with _lock:
            return json.dumps(asdict(_state))

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            try:
                payload = _get_metrics_json()
            except Exception:
                self.send_error(500)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))

    def _start_server() -> None:
        try:
            server = http.server.HTTPServer(("0.0.0.0", metrics_port), _Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"[Kubeflow] Metrics server started on port {metrics_port}", flush=True)
        except OSError as e:
            print(f"[Kubeflow] Warning: Failed to start metrics server: {e}", flush=True)

    def _write_termination_log() -> None:
        try:
            with open("/dev/termination-log", "w") as f:
                f.write(_get_metrics_json())
        except Exception:
            pass

    class MetricsTrackingLoader:
        """Wraps a DataLoader to update progression metrics after each batch."""

        def __init__(
            self,
            loader,
            total_steps: int,
            num_epochs: int,
            start_time: float,
            epoch_offset: int = 0,
        ):
            self.loader = loader
            self.total_steps = total_steps
            self.num_epochs = num_epochs
            self.start_time = start_time
            self.global_step = epoch_offset * len(loader)
            self.batch_sampler = getattr(loader, "batch_sampler", None)

        def __iter__(self):
            for batch in self.loader:
                yield batch
                self.global_step += 1
                elapsed = time.time() - self.start_time
                pct = int(self.global_step / self.total_steps * 100) if self.total_steps > 0 else 0
                remaining = None
                if self.global_step > 0 and elapsed > 0 and self.global_step < self.total_steps:
                    remaining = int(
                        elapsed / self.global_step * (self.total_steps - self.global_step)
                    )
                elif self.global_step >= self.total_steps:
                    remaining = 0
                _update_metrics(
                    {
                        "currentStep": self.global_step,
                        "totalSteps": self.total_steps,
                        "progressPercentage": min(pct, 99),
                        "estimatedRemainingSeconds": remaining,
                    }
                )

        def __len__(self):
            return len(self.loader)

    return _update_metrics, _start_server, _write_termination_log, MetricsTrackingLoader


def _speculator_train_only(
    verifier_model: str,
    speculator_type: str,
    hidden_states_path: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    lr: float,
    total_seq_len: int,
    draft_vocab_size: int | None,
    num_layers: int,
    hidden_states_dtype: str,
    trust_remote_code: bool,
    ttt_steps: int = 3,
    norm_before_residual: bool = True,
    norm_before_fc: bool = False,
    embed_requires_grad: bool = False,
    from_pretrained: str | None = None,
    log_freq: int = 1,
    checkpoint_freq: float = 1.0,
    scheduler_type: str = "linear",
    metrics_port: int = 0,
) -> None:
    """Train an Eagle3 draft model from pre-generated hidden states.

    Injected into training pods via inspect.getsource().
    """
    import os
    from pathlib import Path
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if metrics_port > 0 and local_rank == 0:
        _update_metrics, _start_server, _write_termination, MetricsTrackingLoader = (  # noqa: N806
            _create_speculator_progression_server(metrics_port)
        )
        _start_server()
        _update_metrics({"currentPhase": "initializing", "totalEpochs": epochs})

    os.makedirs(output_dir, exist_ok=True)

    if local_rank == 0:
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}", flush=True)
        print(f"Distributed: {is_distributed} (world_size={world_size})", flush=True)

    verifier_config = AutoConfig.from_pretrained(
        verifier_model, trust_remote_code=trust_remote_code
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    target_vocab_size = verifier_config.vocab_size

    token_freq_path = Path(data_path) / "token_freq.pt"
    resolved_draft_vocab = draft_vocab_size or min(8192, target_vocab_size)

    from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

    token_freq_dict = torch.load(str(token_freq_path), weights_only=True)
    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=resolved_draft_vocab,
        target_vocab_size=target_vocab_size,
    )
    d2t_path = Path(data_path) / "d2t.npy"
    t2d_path = Path(data_path) / "t2d.npy"
    np.save(str(d2t_path), d2t.cpu().numpy())
    np.save(str(t2d_path), t2d.cpu().numpy())

    from speculators.model import SpeculatorModel

    draft_model_cls = SpeculatorModel.registry[speculator_type]
    model_kwargs = {
        "draft_vocab_size": resolved_draft_vocab,
        "num_layers": num_layers,
        "norm_before_residual": norm_before_residual,
        "norm_before_fc": norm_before_fc,
        "embed_requires_grad": embed_requires_grad,
        "ttt_steps": ttt_steps,
        "verifier_name_or_path": verifier_model,
        "d2t": d2t,
        "t2d": t2d,
    }
    if from_pretrained:
        model_kwargs["from_pretrained"] = from_pretrained
    draft_model = draft_model_cls.from_training_args(verifier_config, **model_kwargs)
    print(f"Draft model created (type={speculator_type}, layers={num_layers})", flush=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    from speculators.train.data import ArrowDataset, create_collate_fn

    dataset = ArrowDataset(
        max_len=total_seq_len,
        datapath=data_path,
        hidden_states_path=hidden_states_path,
        on_missing="skip",
        split_ratio=0.9,
        hidden_states_dtype=dtype_map[hidden_states_dtype],
    )
    collate_fn = create_collate_fn(total_seq_len, verifier_config.hidden_size)
    sampler = DistributedSampler(dataset) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    total_steps = len(loader) * epochs
    if local_rank == 0:
        print(f"Dataset: {len(dataset)} samples", flush=True)
        if metrics_port > 0:
            _update_metrics({"currentPhase": "training", "totalSteps": total_steps})
            _train_start = time.time()
            loader = MetricsTrackingLoader(loader, total_steps, epochs, _train_start)

    from speculators.models.eagle3.data import shift_batch
    from speculators.train.trainer import Trainer, TrainerConfig

    config = TrainerConfig(
        num_epochs=epochs,
        save_path=output_dir,
        lr=lr,
        scheduler_type=scheduler_type,
        checkpoint_freq=checkpoint_freq,
        log_freq=log_freq,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"shift_fn": shift_batch},
    )

    trainer = Trainer(draft_model, config, loader)
    if local_rank == 0:
        print("Starting training...", flush=True)
    trainer.run_training()
    if local_rank == 0:
        print(f"Training complete. Checkpoints at: {output_dir}", flush=True)
        if metrics_port > 0:
            _update_metrics(
                {
                    "progressPercentage": 100,
                    "estimatedRemainingSeconds": 0,
                    "currentPhase": "complete",
                }
            )
            _write_termination()


def _speculator_data_only(
    verifier_model: str,
    data_output_dir: str,
    hidden_states_dir: str,
    dataset_name: str,
    max_samples: int | None,
    total_seq_len: int,
    trust_remote_code: bool,
    vllm_port: int = 8234,
    extraction_script_path: str = "/tmp/data_generation_offline.py",
    regenerate_responses: bool = False,
    response_gen_script_path: str = "/tmp/response_regeneration.py",
    datagen_concurrency: int = 4,
    metrics_port: int = 0,
) -> None:
    """Extract hidden states from a verifier model via vLLM sidecar.

    Expects vLLM to be running on localhost (port set by vllm_port param,
    started by the sidecar container in the CTR). Injected via inspect.getsource().
    """
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import urllib.request

    if metrics_port > 0:
        _update_metrics, _start_server, _write_termination, _ = (
            _create_speculator_progression_server(metrics_port)
        )
        _start_server()
        _update_metrics({"currentPhase": "waiting_for_vllm", "progressPercentage": 0})

    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(hidden_states_dir, exist_ok=True)

    # Step 0: Wait for vLLM sidecar (needed before response gen or extraction)
    print("=== Waiting for vLLM sidecar ===", flush=True)
    for i in range(240):
        try:
            urllib.request.urlopen(f"http://localhost:{vllm_port}/health", timeout=2)
            print(f"vLLM ready after {i * 5}s", flush=True)
            break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("vLLM sidecar did not start within 1200s")

    # Step 0.5: Response regeneration (optional)
    if regenerate_responses:
        print("=== Step 0.5: Response regeneration ===", flush=True)
        regen_dataset_map = {
            "sharegpt": "magpie",
            "magpie": "magpie",
            "ultrachat": "ultrachat",
            "gsm8k": "gsm8k",
        }
        regen_dataset = regen_dataset_map.get(dataset_name, "magpie")
        regen_output = str(Path(data_output_dir) / "regenerated_responses.jsonl")
        regen_cmd = [
            sys.executable,
            response_gen_script_path,
            "--endpoint",
            f"http://localhost:{vllm_port}/v1/chat/completions",
            "--dataset",
            regen_dataset,
            "--outfile",
            regen_output,
        ]
        if max_samples is not None:
            regen_cmd.extend(["--limit", str(max_samples)])
        print(f"Running: {' '.join(regen_cmd)}", flush=True)
        subprocess.run(regen_cmd, check=True)
        print(f"Responses saved to {regen_output}", flush=True)
        dataset_name = regen_output

    if metrics_port > 0:
        _update_metrics({"currentPhase": "preprocessing", "progressPercentage": 5})
    print("=== Step 1: Preprocessing ===", flush=True)
    from speculators.data_generation.preprocessing import load_and_preprocess_dataset

    preprocess_kwargs: dict = {
        "target_model_path": verifier_model,
        "train_data_paths": [dataset_name],
        "seq_length": total_seq_len,
        "token_freq_path": str(Path(data_output_dir) / "token_freq.pt"),
    }
    if max_samples is not None:
        preprocess_kwargs["max_samples"] = max_samples

    dataset, _ = load_and_preprocess_dataset(**preprocess_kwargs)
    dataset.save_to_disk(data_output_dir)
    print("Arrow dataset saved", flush=True)

    if metrics_port > 0:
        _update_metrics({"currentPhase": "extracting", "progressPercentage": 10})
    print("=== Step 2: Extracting hidden states ===", flush=True)
    cmd = [
        sys.executable,
        extraction_script_path,
        "--preprocessed-data",
        data_output_dir,
        "--endpoint",
        f"http://localhost:{vllm_port}/v1",
        "--output",
        hidden_states_dir,
        "--concurrency",
        str(datagen_concurrency),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    subprocess.run(cmd, check=True)
    print(f"Hidden states saved to {hidden_states_dir}", flush=True)
    print("Data extraction complete!", flush=True)
    if metrics_port > 0:
        _update_metrics(
            {"progressPercentage": 100, "estimatedRemainingSeconds": 0, "currentPhase": "complete"}
        )
        _write_termination()


def _speculator_offline(
    verifier_model: str,
    speculator_type: str,
    dataset_name: str,
    output_dir: str,
    vllm_endpoint: str,
    epochs: int,
    lr: float,
    total_seq_len: int,
    draft_vocab_size: int | None,
    num_layers: int,
    max_samples: int | None,
    hidden_states_dtype: str,
    trust_remote_code: bool,
    data_output_dir: str | None,
    extraction_script_path: str = "/tmp/data_generation_offline.py",
    datagen_concurrency: int = 4,
    norm_before_residual: bool = True,
    ttt_steps: int = 3,
    norm_before_fc: bool = False,
    embed_requires_grad: bool = False,
    scheduler_type: str = "linear",
    checkpoint_freq: float = 1.0,
    log_freq: int = 1,
    from_pretrained: str | None = None,
    metrics_port: int = 0,
) -> None:
    """End-to-end offline training: extract hidden states via external vLLM, then train.

    Injected via inspect.getsource().
    """
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import urllib.request

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if metrics_port > 0 and local_rank == 0:
        _update_metrics, _start_server, _write_termination, MetricsTrackingLoader = (  # noqa: N806
            _create_speculator_progression_server(metrics_port)
        )
        _start_server()
        _update_metrics({"currentPhase": "initializing", "totalEpochs": epochs})

    if data_output_dir is None:
        data_output_dir = str(Path(output_dir) / "data")
    hs_dir = str(Path(data_output_dir) / "hidden_states")

    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(hs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Preprocess
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "preprocessing", "progressPercentage": 5})
    print("=== Step 1: Preprocessing ===", flush=True)
    from speculators.data_generation.preprocessing import load_and_preprocess_dataset

    preprocess_kwargs: dict = {
        "target_model_path": verifier_model,
        "train_data_paths": [dataset_name],
        "seq_length": total_seq_len,
        "token_freq_path": str(Path(data_output_dir) / "token_freq.pt"),
    }
    if max_samples is not None:
        preprocess_kwargs["max_samples"] = max_samples

    dataset_obj, _ = load_and_preprocess_dataset(**preprocess_kwargs)
    dataset_obj.save_to_disk(data_output_dir)
    print("Arrow dataset saved", flush=True)

    # Step 2: Check vLLM
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "checking_vllm", "progressPercentage": 8})
    print("=== Step 2: Checking vLLM ===", flush=True)
    health = vllm_endpoint.replace("/v1", "/health")
    for _i in range(60):
        try:
            urllib.request.urlopen(health, timeout=2)
            print("vLLM ready", flush=True)
            break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("vLLM endpoint not reachable")

    # Step 3: Extract hidden states
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "extracting", "progressPercentage": 10})
    print("=== Step 3: Extracting ===", flush=True)
    cmd = [
        sys.executable,
        extraction_script_path,
        "--preprocessed-data",
        data_output_dir,
        "--endpoint",
        vllm_endpoint,
        "--output",
        hs_dir,
        "--concurrency",
        str(datagen_concurrency),
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    subprocess.run(cmd, check=True)
    print("Hidden states extracted", flush=True)

    # Step 4: Train
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "training", "progressPercentage": 15})
    print("=== Step 4: Training ===", flush=True)
    verifier_config = AutoConfig.from_pretrained(
        verifier_model, trust_remote_code=trust_remote_code
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    target_vocab_size = verifier_config.vocab_size

    token_freq_dict = torch.load(str(Path(data_output_dir) / "token_freq.pt"), weights_only=True)
    resolved_draft_vocab = draft_vocab_size or min(8192, target_vocab_size)
    from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=resolved_draft_vocab,
        target_vocab_size=target_vocab_size,
    )
    np.save(str(Path(data_output_dir) / "d2t.npy"), d2t.cpu().numpy())
    np.save(str(Path(data_output_dir) / "t2d.npy"), t2d.cpu().numpy())

    from speculators.model import SpeculatorModel

    draft_model_cls = SpeculatorModel.registry[speculator_type]
    from_training_kwargs: dict = {
        "draft_vocab_size": resolved_draft_vocab,
        "num_layers": num_layers,
        "norm_before_residual": norm_before_residual,
        "norm_before_fc": norm_before_fc,
        "embed_requires_grad": embed_requires_grad,
        "ttt_steps": ttt_steps,
        "verifier_name_or_path": verifier_model,
        "d2t": d2t,
        "t2d": t2d,
    }
    if from_pretrained is not None:
        from_training_kwargs["from_pretrained"] = from_pretrained
    draft_model = draft_model_cls.from_training_args(verifier_config, **from_training_kwargs)
    print(f"Draft model created (type={speculator_type}, layers={num_layers})", flush=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    from speculators.train.data import ArrowDataset, create_collate_fn

    train_dataset = ArrowDataset(
        max_len=total_seq_len,
        datapath=data_output_dir,
        hidden_states_path=hs_dir,
        on_missing="skip",
        split_ratio=0.9,
        hidden_states_dtype=dtype_map[hidden_states_dtype],
    )
    collate_fn = create_collate_fn(total_seq_len, verifier_config.hidden_size)
    sampler = DistributedSampler(train_dataset) if is_distributed else None
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    total_steps = len(loader) * epochs
    if local_rank == 0:
        print(f"Dataset: {len(train_dataset)} samples", flush=True)
        if metrics_port > 0:
            _update_metrics({"currentPhase": "training", "totalSteps": total_steps})
            _train_start = time.time()
            loader = MetricsTrackingLoader(loader, total_steps, epochs, _train_start)

    from speculators.models.eagle3.data import shift_batch
    from speculators.train.trainer import Trainer, TrainerConfig

    trainer_config = TrainerConfig(
        num_epochs=epochs,
        save_path=output_dir,
        lr=lr,
        scheduler_type=scheduler_type,
        checkpoint_freq=checkpoint_freq,
        log_freq=log_freq,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"shift_fn": shift_batch},
    )

    trainer = Trainer(draft_model, trainer_config, loader)
    if local_rank == 0:
        print("Starting training...", flush=True)
    trainer.run_training()
    if local_rank == 0:
        print(f"Offline mode complete. Checkpoints at: {output_dir}", flush=True)
        if metrics_port > 0:
            _update_metrics(
                {
                    "progressPercentage": 100,
                    "estimatedRemainingSeconds": 0,
                    "currentPhase": "complete",
                }
            )
            _write_termination()


def _speculator_online(
    verifier_model: str,
    speculator_type: str,
    dataset_name: str,
    output_dir: str,
    epochs: int,
    lr: float,
    total_seq_len: int,
    draft_vocab_size: int | None,
    num_layers: int,
    max_samples: int | None,
    hidden_states_dtype: str,
    trust_remote_code: bool,
    vllm_port: int = 8234,
    norm_before_residual: bool = True,
    ttt_steps: int = 3,
    norm_before_fc: bool = False,
    embed_requires_grad: bool = False,
    scheduler_type: str = "linear",
    checkpoint_freq: float = 1.0,
    log_freq: int = 1,
    from_pretrained: str | None = None,
    metrics_port: int = 0,
) -> None:
    """Online training: vLLM sidecar generates hidden states on-the-fly during training.

    Expects vLLM to be running on localhost (port set by vllm_port param)
    with extraction config. Injected via inspect.getsource().
    """
    import os
    from pathlib import Path
    import sys
    import time
    import urllib.request

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if metrics_port > 0 and local_rank == 0:
        _update_metrics, _start_server, _write_termination, MetricsTrackingLoader = (  # noqa: N806
            _create_speculator_progression_server(metrics_port)
        )
        _start_server()
        _update_metrics({"currentPhase": "initializing", "totalEpochs": epochs})

    data_dir = str(Path(output_dir) / "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Step 1: Preprocess
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "preprocessing", "progressPercentage": 5})
    print("=== Step 1: Preprocessing ===", flush=True)
    from speculators.data_generation.preprocessing import load_and_preprocess_dataset

    preprocess_kwargs: dict = {
        "target_model_path": verifier_model,
        "train_data_paths": [dataset_name],
        "seq_length": total_seq_len,
        "token_freq_path": str(Path(data_dir) / "token_freq.pt"),
    }
    if max_samples is not None:
        preprocess_kwargs["max_samples"] = max_samples

    dataset_obj, _ = load_and_preprocess_dataset(**preprocess_kwargs)
    dataset_obj.save_to_disk(data_dir)
    print("Arrow dataset saved", flush=True)

    # Step 2: Wait for vLLM sidecar
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "waiting_for_vllm", "progressPercentage": 8})
    print("=== Step 2: Waiting for vLLM ===", flush=True)
    for i in range(240):
        try:
            urllib.request.urlopen(f"http://localhost:{vllm_port}/health", timeout=2)
            print(f"vLLM ready after {i * 5}s", flush=True)
            break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("vLLM sidecar did not start within 1200s")

    # Step 3: Setup
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "setting_up_model", "progressPercentage": 10})
    print("=== Step 3: Setting up model ===", flush=True)
    verifier_config = AutoConfig.from_pretrained(
        verifier_model, trust_remote_code=trust_remote_code
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    token_freq_dict = torch.load(str(Path(data_dir) / "token_freq.pt"), weights_only=True)
    resolved_draft_vocab = draft_vocab_size or min(8192, verifier_config.vocab_size)
    from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=resolved_draft_vocab,
        target_vocab_size=verifier_config.vocab_size,
    )
    np.save(str(Path(data_dir) / "d2t.npy"), d2t.cpu().numpy())
    np.save(str(Path(data_dir) / "t2d.npy"), t2d.cpu().numpy())

    from speculators.model import SpeculatorModel

    draft_model_cls = SpeculatorModel.registry[speculator_type]
    from_training_kwargs: dict = {
        "draft_vocab_size": resolved_draft_vocab,
        "num_layers": num_layers,
        "norm_before_residual": norm_before_residual,
        "norm_before_fc": norm_before_fc,
        "embed_requires_grad": embed_requires_grad,
        "ttt_steps": ttt_steps,
        "verifier_name_or_path": verifier_model,
        "d2t": d2t,
        "t2d": t2d,
    }
    if from_pretrained is not None:
        from_training_kwargs["from_pretrained"] = from_pretrained
    draft_model = draft_model_cls.from_training_args(verifier_config, **from_training_kwargs)
    print(f"Draft model created (type={speculator_type}, layers={num_layers})", flush=True)

    # Step 4: Online training
    if metrics_port > 0 and local_rank == 0:
        _update_metrics({"currentPhase": "training", "progressPercentage": 15})
    print("=== Step 4: Online training ===", flush=True)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    from speculators.train.data import ArrowDataset, create_collate_fn

    ds = ArrowDataset(
        max_len=total_seq_len,
        datapath=data_dir,
        hidden_states_path=str(Path(output_dir) / "data" / "hidden_states"),
        on_missing="generate",
        vllm_endpoint=f"http://localhost:{vllm_port}/v1",
        split_ratio=0.9,
        hidden_states_dtype=dtype_map[hidden_states_dtype],
    )
    collate_fn = create_collate_fn(total_seq_len, verifier_config.hidden_size)
    sampler = DistributedSampler(ds) if is_distributed else None
    loader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    total_steps = len(loader) * epochs
    if local_rank == 0:
        print(f"Dataset: {len(ds)} samples (online)", flush=True)
        if metrics_port > 0:
            _update_metrics({"currentPhase": "training", "totalSteps": total_steps})
            _train_start = time.time()
            loader = MetricsTrackingLoader(loader, total_steps, epochs, _train_start)

    from speculators.models.eagle3.data import shift_batch
    from speculators.train.trainer import Trainer, TrainerConfig

    config = TrainerConfig(
        num_epochs=epochs,
        save_path=output_dir,
        lr=lr,
        scheduler_type=scheduler_type,
        checkpoint_freq=checkpoint_freq,
        log_freq=log_freq,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"shift_fn": shift_batch},
    )

    trainer = Trainer(draft_model, config, loader)
    if local_rank == 0:
        print("Starting online training...", flush=True)
    trainer.run_training()
    if local_rank == 0:
        print(f"Online mode complete. Checkpoints at: {output_dir}", flush=True)
        if metrics_port > 0:
            _update_metrics(
                {
                    "progressPercentage": 100,
                    "estimatedRemainingSeconds": 0,
                    "currentPhase": "complete",
                }
            )
            _write_termination()


# =============================================================================
# Script rendering
# =============================================================================

_MODE_FUNCTIONS = {
    SpeculatorMode.TRAIN_ONLY: _speculator_train_only,
    SpeculatorMode.DATA_ONLY: _speculator_data_only,
    SpeculatorMode.OFFLINE: _speculator_offline,
    SpeculatorMode.ONLINE: _speculator_online,
}


def _load_bundled_script(script_name: str) -> str:
    """Load a bundled script from speculator_scripts/ directory."""
    from pathlib import Path

    from kubeflow.trainer.rhai.constants import SPECULATOR_SCRIPTS_DIR

    script_path = Path(__file__).parent / SPECULATOR_SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Bundled script '{script_name}' not found at {script_path}. "
            f"Expected in {SPECULATOR_SCRIPTS_DIR}/ directory."
        )
    return script_path.read_text()


def _bundled_script_preamble(script_name: str, dest_path: str) -> str:
    """Generate preamble code that writes a bundled script to disk via base64."""
    import base64

    content = _load_bundled_script(script_name)
    b64 = base64.b64encode(content.encode()).decode()
    return (
        f"import base64 as _b64\n"
        f"with open({dest_path!r}, 'w') as _f:\n"
        f"    _f.write(_b64.b64decode({b64!r}).decode())\n"
        "\n"
    )


def _render_speculator_mode_script(trainer: SpeculatorTrainer) -> str:
    """Generate a self-contained training script from mode and parameters."""
    from kubeflow.trainer.rhai.constants import (
        SPECULATOR_DATA_SUBDIR,
        SPECULATOR_HIDDEN_STATES_SUBDIR,
        SPECULATOR_SIDECAR_PORT,
    )
    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    mode_func = _MODE_FUNCTIONS[trainer.mode]
    func_source = inspect.getsource(mode_func)
    func_source = textwrap.dedent(func_source)

    resolved_output_dir = trainer.output_dir or "/tmp/speculator_output"
    if trainer.output_dir:
        resolved_path, _ = parse_output_dir_uri(trainer.output_dir)
        if resolved_path:
            resolved_output_dir = resolved_path

    resolved_hs_path = trainer.hidden_states_path
    if trainer.hidden_states_path and trainer.hidden_states_path.startswith("pvc://"):
        hs_resolved, _ = parse_output_dir_uri(trainer.hidden_states_path)
        if hs_resolved:
            resolved_hs_path = hs_resolved

    data_output_dir = resolved_output_dir + "/" + SPECULATOR_DATA_SUBDIR
    cfg = trainer.config
    metrics_port = trainer.metrics_port if trainer.enable_progression_tracking else 0

    if trainer.mode == SpeculatorMode.TRAIN_ONLY:
        call = (
            f"{mode_func.__name__}(\n"
            f"    verifier_model={trainer.verifier_model!r},\n"
            f"    speculator_type={trainer.speculator_type.value!r},\n"
            f"    hidden_states_path={resolved_hs_path!r},\n"
            f"    data_path={trainer.data_path!r},\n"
            f"    output_dir={resolved_output_dir!r},\n"
            f"    epochs={trainer.epochs!r},\n"
            f"    lr={trainer.lr!r},\n"
            f"    total_seq_len={trainer.total_seq_len!r},\n"
            f"    draft_vocab_size={trainer.draft_vocab_size!r},\n"
            f"    num_layers={cfg.num_layers!r},\n"
            f"    hidden_states_dtype={cfg.hidden_states_dtype!r},\n"
            f"    trust_remote_code={trainer.trust_remote_code!r},\n"
            f"    ttt_steps={cfg.ttt_steps!r},\n"
            f"    norm_before_residual={cfg.norm_before_residual!r},\n"
            f"    norm_before_fc={cfg.norm_before_fc!r},\n"
            f"    embed_requires_grad={cfg.embed_requires_grad!r},\n"
            f"    from_pretrained={cfg.from_pretrained!r},\n"
            f"    log_freq={cfg.log_freq!r},\n"
            f"    checkpoint_freq={cfg.checkpoint_freq!r},\n"
            f"    scheduler_type={cfg.scheduler_type!r},\n"
            f"    metrics_port={metrics_port!r},\n"
            f")\n"
        )

    elif trainer.mode == SpeculatorMode.DATA_ONLY:
        call = (
            f"{mode_func.__name__}(\n"
            f"    verifier_model={trainer.verifier_model!r},\n"
            f"    data_output_dir={data_output_dir!r},\n"
            f"    hidden_states_dir={resolved_output_dir + '/' + SPECULATOR_HIDDEN_STATES_SUBDIR!r},\n"
            f"    dataset_name={trainer.dataset_name!r},\n"
            f"    max_samples={trainer.max_samples!r},\n"
            f"    total_seq_len={trainer.total_seq_len!r},\n"
            f"    trust_remote_code={trainer.trust_remote_code!r},\n"
            f"    vllm_port={SPECULATOR_SIDECAR_PORT!r},\n"
            f"    regenerate_responses={trainer.regenerate_responses!r},\n"
            f"    datagen_concurrency={cfg.datagen_concurrency!r},\n"
            f"    metrics_port={metrics_port!r},\n"
            f")\n"
        )

    elif trainer.mode == SpeculatorMode.OFFLINE:
        call = (
            f"{mode_func.__name__}(\n"
            f"    verifier_model={trainer.verifier_model!r},\n"
            f"    speculator_type={trainer.speculator_type.value!r},\n"
            f"    dataset_name={trainer.dataset_name!r},\n"
            f"    output_dir={resolved_output_dir!r},\n"
            f"    vllm_endpoint={trainer.vllm_endpoint!r},\n"
            f"    epochs={trainer.epochs!r},\n"
            f"    lr={trainer.lr!r},\n"
            f"    total_seq_len={trainer.total_seq_len!r},\n"
            f"    draft_vocab_size={trainer.draft_vocab_size!r},\n"
            f"    num_layers={cfg.num_layers!r},\n"
            f"    max_samples={trainer.max_samples!r},\n"
            f"    hidden_states_dtype={cfg.hidden_states_dtype!r},\n"
            f"    trust_remote_code={trainer.trust_remote_code!r},\n"
            f"    data_output_dir={data_output_dir!r},\n"
            f"    extraction_script_path='/tmp/data_generation_offline.py',\n"
            f"    datagen_concurrency={cfg.datagen_concurrency!r},\n"
            f"    norm_before_residual={cfg.norm_before_residual!r},\n"
            f"    ttt_steps={cfg.ttt_steps!r},\n"
            f"    norm_before_fc={cfg.norm_before_fc!r},\n"
            f"    embed_requires_grad={cfg.embed_requires_grad!r},\n"
            f"    scheduler_type={cfg.scheduler_type!r},\n"
            f"    checkpoint_freq={cfg.checkpoint_freq!r},\n"
            f"    log_freq={cfg.log_freq!r},\n"
            f"    from_pretrained={cfg.from_pretrained!r},\n"
            f"    metrics_port={metrics_port!r},\n"
            f")\n"
        )

    elif trainer.mode == SpeculatorMode.ONLINE:
        call = (
            f"{mode_func.__name__}(\n"
            f"    verifier_model={trainer.verifier_model!r},\n"
            f"    speculator_type={trainer.speculator_type.value!r},\n"
            f"    dataset_name={trainer.dataset_name!r},\n"
            f"    output_dir={resolved_output_dir!r},\n"
            f"    epochs={trainer.epochs!r},\n"
            f"    lr={trainer.lr!r},\n"
            f"    total_seq_len={trainer.total_seq_len!r},\n"
            f"    draft_vocab_size={trainer.draft_vocab_size!r},\n"
            f"    num_layers={cfg.num_layers!r},\n"
            f"    max_samples={trainer.max_samples!r},\n"
            f"    hidden_states_dtype={cfg.hidden_states_dtype!r},\n"
            f"    trust_remote_code={trainer.trust_remote_code!r},\n"
            f"    vllm_port={SPECULATOR_SIDECAR_PORT!r},\n"
            f"    norm_before_residual={cfg.norm_before_residual!r},\n"
            f"    ttt_steps={cfg.ttt_steps!r},\n"
            f"    norm_before_fc={cfg.norm_before_fc!r},\n"
            f"    embed_requires_grad={cfg.embed_requires_grad!r},\n"
            f"    scheduler_type={cfg.scheduler_type!r},\n"
            f"    checkpoint_freq={cfg.checkpoint_freq!r},\n"
            f"    log_freq={cfg.log_freq!r},\n"
            f"    from_pretrained={cfg.from_pretrained!r},\n"
            f"    metrics_port={metrics_port!r},\n"
            f")\n"
        )

    else:
        raise ValueError(f"Unsupported mode: {trainer.mode}")

    pip_preamble = (
        "import subprocess, sys\n"
        "subprocess.run(\n"
        "    [sys.executable, '-m', 'pip', 'install', '--upgrade',\n"
        "     '--force-reinstall', '--no-deps', '--no-cache-dir',\n"
        "     '--index-url', 'https://pypi.org/simple/', 'speculators==0.6.0'],\n"
        "    capture_output=True, text=True,\n"
        ")\n\n"
    )

    progression_source = ""
    if trainer.enable_progression_tracking:
        progression_source = textwrap.dedent(
            inspect.getsource(_create_speculator_progression_server)
        )
        progression_source += "\n\n"

    preamble = ""
    if trainer.mode in (SpeculatorMode.DATA_ONLY, SpeculatorMode.OFFLINE):
        from kubeflow.trainer.rhai.constants import (
            SPECULATOR_EXTRACTION_SCRIPT,
            SPECULATOR_RESPONSE_GEN_SCRIPT,
        )

        preamble = _bundled_script_preamble(
            SPECULATOR_EXTRACTION_SCRIPT, "/tmp/data_generation_offline.py"
        )

        if trainer.mode == SpeculatorMode.DATA_ONLY and trainer.regenerate_responses:
            preamble += _bundled_script_preamble(
                SPECULATOR_RESPONSE_GEN_SCRIPT, "/tmp/response_regeneration.py"
            )

    return f"{pip_preamble}{progression_source}{func_source}\n{preamble}{call}"


def get_trainer_cr_from_speculator_trainer(
    runtime: types.Runtime,
    trainer: SpeculatorTrainer,
    initializer: types.Initializer | None = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD spec for SpeculatorTrainer."""
    if trainer.mode in (SpeculatorMode.TRAIN_ONLY, SpeculatorMode.OFFLINE, SpeculatorMode.ONLINE):
        runtime.trainer.set_command(constants.TORCH_COMMAND)
    else:
        runtime.trainer.set_command(constants.DEFAULT_COMMAND)

    func_code = _render_speculator_mode_script(trainer)
    func_file = "speculator_training.py"

    trainer_crd = models.TrainerV1alpha1Trainer()

    if trainer.resources_per_node:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            trainer.resources_per_node
        )
    elif trainer.mode in (SpeculatorMode.TRAIN_ONLY, SpeculatorMode.OFFLINE, SpeculatorMode.ONLINE):
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            {"nvidia.com/gpu": trainer.training_gpu_count}
        )

    install_snippet = ""
    if trainer.packages_to_install:
        install_snippet = k8s_utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
        )

    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_snippet:
                exec_script = install_snippet + exec_script
            command.append(exec_script)
        else:
            command.append(c)
    trainer_crd.command = command

    env_vars = {}
    if trainer.env:
        env_vars.update(trainer.env)
    env_vars.setdefault("HF_HUB_OFFLINE", "0")
    env_vars.setdefault("HF_HOME", "/tmp/hf_cache")
    env_vars.setdefault("TORCH_DYNAMO_DISABLE", "1")

    trainer_crd.env = [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in env_vars.items()]

    return trainer_crd
