import pytest

from kubeflow.trainer.rhai.speculator import (
    SpeculatorConfig,
    SpeculatorMode,
    SpeculatorTrainer,
    SpeculatorType,
    _render_speculator_mode_script,
)

PVC_OUTPUT = "pvc://shared/speculator/output"
PVC_HS = "pvc://shared/speculator/hidden_states"


class TestSpeculatorTrainerInit:
    def test_train_only_defaults(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow_dataset",
            output_dir=PVC_OUTPUT,
        )
        assert trainer.mode == SpeculatorMode.TRAIN_ONLY
        assert trainer.speculator_type == SpeculatorType.EAGLE3
        assert trainer.epochs == 3
        assert trainer.config.hidden_states_dtype == "bfloat16"

    def test_data_only_defaults(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        assert trainer.mode == SpeculatorMode.DATA_ONLY
        assert trainer.dataset_name == "sharegpt"

    def test_offline_defaults(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.OFFLINE,
            verifier_model="Qwen/Qwen3-8B",
            vllm_endpoint="http://vllm:8000/v1",
            output_dir=PVC_OUTPUT,
        )
        assert trainer.mode == SpeculatorMode.OFFLINE

    def test_online_defaults(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.ONLINE,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        assert trainer.mode == SpeculatorMode.ONLINE

    def test_output_dir_required(self):
        with pytest.raises(ValueError, match="output_dir.*required"):
            SpeculatorTrainer(
                mode=SpeculatorMode.DATA_ONLY,
                verifier_model="Qwen/Qwen3-8B",
            )

    def test_s3_output_dir_rejected(self):
        with pytest.raises(ValueError, match="S3.*not supported"):
            SpeculatorTrainer(
                mode=SpeculatorMode.DATA_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                output_dir="s3://bucket/path",
            )

    def test_s3_hidden_states_rejected(self):
        with pytest.raises(ValueError, match="S3.*not supported"):
            SpeculatorTrainer(
                mode=SpeculatorMode.TRAIN_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                hidden_states_path="s3://bucket/hs",
                data_path="/data/arrow",
                output_dir=PVC_OUTPUT,
            )

    def test_different_pvc_rejected(self):
        with pytest.raises(ValueError, match="same PVC"):
            SpeculatorTrainer(
                mode=SpeculatorMode.TRAIN_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                hidden_states_path="pvc://other-pvc/hs",
                data_path="/data/arrow",
                output_dir=PVC_OUTPUT,
            )

    def test_same_pvc_allowed(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path="pvc://shared/hs",
            data_path="/data/arrow",
            output_dir="pvc://shared/checkpoints",
        )
        assert trainer.hidden_states_path == "pvc://shared/hs"

    def test_train_only_missing_hidden_states(self):
        with pytest.raises(ValueError, match="hidden_states_path"):
            SpeculatorTrainer(
                mode=SpeculatorMode.TRAIN_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                data_path="/data/arrow",
                output_dir=PVC_OUTPUT,
            )

    def test_train_only_missing_data_path(self):
        with pytest.raises(ValueError, match="data_path"):
            SpeculatorTrainer(
                mode=SpeculatorMode.TRAIN_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                hidden_states_path=PVC_HS,
                output_dir=PVC_OUTPUT,
            )

    def test_offline_missing_vllm_endpoint(self):
        with pytest.raises(ValueError, match="vllm_endpoint"):
            SpeculatorTrainer(
                mode=SpeculatorMode.OFFLINE,
                verifier_model="Qwen/Qwen3-8B",
                output_dir=PVC_OUTPUT,
            )

    def test_invalid_speculator_type(self):
        with pytest.raises(ValueError, match="speculator_type"):
            SpeculatorTrainer(
                mode=SpeculatorMode.DATA_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                output_dir=PVC_OUTPUT,
                speculator_type="invalid_not_enum",
            )

    def test_invalid_dtype_via_config(self):
        with pytest.raises(ValueError, match="hidden_states_dtype"):
            SpeculatorTrainer(
                mode=SpeculatorMode.DATA_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                output_dir=PVC_OUTPUT,
                config=SpeculatorConfig(hidden_states_dtype="float64"),
            )

    def test_config_defaults_applied(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        assert trainer.config is not None
        assert trainer.config.num_layers == 1
        assert trainer.config.ttt_steps == 3
        assert trainer.config.scheduler_type == "linear"

    def test_custom_config(self):
        cfg = SpeculatorConfig(num_layers=2, ttt_steps=5, loss_fn="ce")
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
            config=cfg,
        )
        assert trainer.config.num_layers == 2
        assert trainer.config.ttt_steps == 5
        assert trainer.config.loss_fn == "ce"

    def test_pvc_uri_normalized(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir="pvc://shared/path/",
        )
        assert trainer.output_dir == "pvc://shared/path"


class TestSpeculatorModeScriptRendering:
    def test_train_only_compiles(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        compile(code, "train_only.py", "exec")

    def test_train_only_has_correct_imports(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        assert "SpeculatorModel" in code
        assert "ArrowDataset" in code
        assert "Trainer" in code

    def test_train_only_resolves_pvc_paths(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path="pvc://shared/speculator/hs",
            data_path="/data/arrow",
            output_dir="pvc://shared/speculator/checkpoints",
        )
        code = _render_speculator_mode_script(trainer)
        assert "/mnt/kubeflow-checkpoints/speculator/checkpoints" in code
        assert "/mnt/kubeflow-checkpoints/speculator/hs" in code
        assert "pvc://" not in code

    def test_data_only_compiles(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        compile(code, "data_only.py", "exec")

    def test_data_only_has_preprocessing(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        assert "load_and_preprocess_dataset" in code
        assert "localhost:" in code

    def test_offline_compiles(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.OFFLINE,
            verifier_model="Qwen/Qwen3-8B",
            vllm_endpoint="http://vllm:8000/v1",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        compile(code, "offline.py", "exec")

    def test_offline_has_both_phases(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.OFFLINE,
            verifier_model="Qwen/Qwen3-8B",
            vllm_endpoint="http://vllm:8000/v1",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        assert "data_generation_offline" in code
        assert "Trainer" in code
        assert "run_training" in code

    def test_online_compiles(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.ONLINE,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        compile(code, "online.py", "exec")

    def test_online_has_generate_mode(self):
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.ONLINE,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        code = _render_speculator_mode_script(trainer)
        assert 'on_missing="generate"' in code
        assert "localhost:" in code

    def test_speculator_type_embedded(self):
        for stype in SpeculatorType:
            trainer = SpeculatorTrainer(
                mode=SpeculatorMode.TRAIN_ONLY,
                verifier_model="Qwen/Qwen3-8B",
                hidden_states_path=PVC_HS,
                data_path="/data/arrow",
                output_dir=PVC_OUTPUT,
                speculator_type=stype,
            )
            code = _render_speculator_mode_script(trainer)
            assert f"speculator_type='{stype.value}'" in code


class TestSpeculatorTrainerCRD:
    @staticmethod
    def _make_runtime():
        from kubeflow.trainer.types.types import Runtime, RuntimeTrainer, TrainerType

        rt = RuntimeTrainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="test-image:latest",
            num_nodes=1,
            device="gpu",
            device_count="1",
        )
        rt.set_command(("bash", "-c", "placeholder {func_file} {func_code}"))
        return Runtime(name="test-runtime", trainer=rt)

    def test_crd_generates_script(self):
        runtime = self._make_runtime()
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow",
            output_dir=PVC_OUTPUT,
        )
        from kubeflow.trainer.rhai.speculator import get_trainer_cr_from_speculator_trainer

        crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)
        command_str = " ".join(crd.command)
        assert "SpeculatorModel" in command_str

    def test_crd_includes_default_env(self):
        runtime = self._make_runtime()
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.DATA_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            output_dir=PVC_OUTPUT,
        )
        from kubeflow.trainer.rhai.speculator import get_trainer_cr_from_speculator_trainer

        crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)
        env_names = {e.name for e in crd.env}
        assert "HF_HUB_OFFLINE" in env_names
        assert "HF_HOME" in env_names
        assert "TORCH_DYNAMO_DISABLE" in env_names

    def test_crd_custom_env_merged(self):
        runtime = self._make_runtime()
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow",
            output_dir=PVC_OUTPUT,
            env={"CUSTOM_VAR": "value"},
        )
        from kubeflow.trainer.rhai.speculator import get_trainer_cr_from_speculator_trainer

        crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)
        env_names = {e.name for e in crd.env}
        assert "CUSTOM_VAR" in env_names
        assert "HF_HUB_OFFLINE" in env_names

    def test_crd_single_node(self):
        runtime = self._make_runtime()
        trainer = SpeculatorTrainer(
            mode=SpeculatorMode.TRAIN_ONLY,
            verifier_model="Qwen/Qwen3-8B",
            hidden_states_path=PVC_HS,
            data_path="/data/arrow",
            output_dir=PVC_OUTPUT,
        )
        from kubeflow.trainer.rhai.speculator import get_trainer_cr_from_speculator_trainer

        crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)
        assert crd.num_nodes is None
