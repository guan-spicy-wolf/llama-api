"""Configuration management for llama-api."""
import os
import yaml
from pathlib import Path
from typing import Optional

from .models import ModelConfig, ModelArgs


# Default paths
DEFAULT_CONFIG_DIR = Path("/var/containers/llama-api")
DEFAULT_MODELS_DIR = Path("/var/models")


class Config:
    """Application configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.models_dir = DEFAULT_MODELS_DIR
        self.llama_server = Path.home() / "llama.cpp/rocm/bin/llama-server"
        self.default_port = 8080
        self.ld_library_path: Optional[str] = None
        self._load_main_config()
        self._model_configs: dict[str, ModelConfig] = {}
        self._load_model_configs()

    def _load_main_config(self) -> None:
        """Load main configuration file."""
        config_file = self.config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            if "models_dir" in data:
                self.models_dir = Path(data["models_dir"])
            if "llama_server" in data:
                self.llama_server = Path(data["llama_server"])
            if "default_port" in data:
                self.default_port = data["default_port"]
            if "ld_library_path" in data:
                self.ld_library_path = data["ld_library_path"]

    def _load_model_configs(self) -> None:
        """Load all model configuration files from models.d/."""
        models_d = self.config_dir / "models.d"
        if not models_d.exists():
            return

        for yaml_file in models_d.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data:
                    config = ModelConfig(**data)
                    self._model_configs[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._model_configs.get(name)

    def list_model_configs(self) -> list[ModelConfig]:
        """List all model configurations."""
        return list(self._model_configs.values())

    def save_model_config(self, config: ModelConfig) -> None:
        """Save model configuration to file."""
        models_d = self.config_dir / "models.d"
        models_d.mkdir(parents=True, exist_ok=True)

        config_file = models_d / f"{config.name}.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config.model_dump(mode="python"), f)

        self._model_configs[config.name] = config

    def delete_model_config(self, name: str) -> bool:
        """Delete model configuration."""
        if name not in self._model_configs:
            return False

        config_file = self.config_dir / "models.d" / f"{name}.yaml"
        if config_file.exists():
            config_file.unlink()

        del self._model_configs[name]
        return True

    def infer_args(self, model_file: str) -> ModelArgs:
        """Infer model arguments from file size."""
        model_path = self.models_dir / model_file
        if not model_path.exists():
            return ModelArgs()

        # Get file size in bytes
        size_bytes = model_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)

        # Infer parameters based on size (rough estimates for GGUF Q4)
        # Q4 models: ~0.3GB per billion parameters
        estimated_params_b = size_gb / 0.3

        # Determine context size
        if estimated_params_b < 10:
            ctx_size = 8192
        elif estimated_params_b < 50:
            ctx_size = 4096
        else:
            ctx_size = 2048

        # Get CPU threads
        cpu_count = os.cpu_count() or 4
        threads = max(1, cpu_count // 2)

        return ModelArgs(
            ctx_size=ctx_size,
            n_gpu_layers=99,  # Full GPU offload for ROCm
            threads=threads,
        )

    def get_effective_args(self, model_name: str) -> ModelArgs:
        """Get effective args: configured args + inferred defaults."""
        config = self.get_model_config(model_name)
        if not config:
            return ModelArgs()

        # Start with inferred args
        inferred = self.infer_args(config.file)

        # Override with configured args
        configured = config.args or ModelArgs()
        effective = inferred.model_dump()

        for key, value in configured.model_dump(exclude_unset=True).items():
            if value is not None:
                effective[key] = value

        return ModelArgs(**effective)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(config_dir: Optional[Path] = None) -> Config:
    """Initialize global config instance."""
    global _config
    _config = Config(config_dir)
    return _config