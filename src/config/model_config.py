"""Saved configurations of the autoencoders, the LSTM, and the echo state network.

Each saved model is a directory named by the hash of its configuration:

    <store>/<hash>/model_config.yaml      the class and its options
    <store>/<hash>/trained_matrices.npz   the trained arrays

`ModelConfig` stores models that implement `Configurable`. `ESNConfig` stores
`ESN_model`, with `esn_config.yaml` in place of `model_config.yaml`.

Typical usage example:

  ae = auto_load_or_create(AE, data=X, training_data_filename="wake.h5", n_latent=8)
  config, path = save_model_to_config(lstm, training_data_filename="wake.h5")
  lstm = load_model_from_config(q=config.to_hash())
  esn = auto_load_or_create_esn(data=Y, dt=0.01, N_units=100)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml

from models.data_driven import ESN_model
from models.data_driven.configurable import Configurable

__all__ = [
    "ESN_STORE",
    "MODEL_STORE",
    "ESNConfig",
    "ModelConfig",
    "auto_load_or_create",
    "auto_load_or_create_esn",
    "find_matching_config",
    "list_saved_configs",
    "list_saved_esn_configs",
    "load_esn_model_from_config",
    "load_model_from_config",
    "save_esn_model_to_config",
    "save_model_to_config",
]

#: Default store, `results/model_configs` in the repository.
MODEL_STORE = Path(__file__).resolve().parents[2] / "results" / "model_configs"
CONFIG_FILE = "model_config.yaml"
TRAINED_FILE = "trained_matrices.npz"
#: Digits floats are rounded to before hashing.
_FLOAT_DIGITS = 12


@dataclass
class ModelConfig:
    """The class and options of a saved model.

    Attributes:
      model_class: Import path of the model class, `module:QualifiedName`.
      options: Constructor options, as `Configurable.config_options` reports
        them.
      training_data_filename: Name of the data the model was trained on.
    """

    model_class: str
    options: dict[str, Any] = field(default_factory=dict[str, Any])
    training_data_filename: str | None = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_model(cls, model: Configurable, training_data_filename: str | None = None) -> ModelConfig:
        """Creates the configuration of a trained model.

        Args:
          model: The model.
          training_data_filename: Name of the data the model was trained on.

        Returns:
          The configuration.

        Raises:
          TypeError: If the model's class cannot be stored.
        """
        model_class = type(model)
        _check_configurable(model_class)
        return cls(_class_path(model_class), model.config_options(), training_data_filename)

    @classmethod
    def from_init_params(
        cls,
        model_class: type[Configurable],
        data: Any = None,
        training_data_filename: str | None = None,
        **options: Any,
    ) -> ModelConfig:
        """Creates the configuration a model built from options would have.

        Args:
          model_class: The model class.
          data: The training data, for options the class infers from it.
          training_data_filename: Name of the training data.
          **options: Constructor options. Missing options take their defaults.

        Returns:
          The configuration.

        Raises:
          TypeError: If `model_class` cannot be stored.
        """
        _check_configurable(model_class)
        return cls(_class_path(model_class), model_class.resolve_options(data, **options), training_data_filename)

    def to_hash(self) -> str:
        """Hashes the configuration.

        Floats are rounded to 12 digits and tuples compared as lists.

        Returns:
          The first 16 hexadecimal digits of the SHA-256 digest.
        """
        text = json.dumps(_canonical(asdict(self)), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── Models ────────────────────────────────────────────────────────────────

    def resolve_class(self) -> type[Configurable]:
        """Imports the model class.

        Returns:
          The class.

        Raises:
          TypeError: If the import path does not name a class that can be
            stored.
        """
        module_name, _, qualname = self.model_class.partition(":")
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, type) or not issubclass(obj, Configurable):
            raise TypeError(f"{self.model_class} is not a Configurable class")
        _check_configurable(obj)
        return obj

    def to_model(self, trained: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Configurable:
        """Rebuilds the trained model.

        Args:
          trained: The trained arrays.
          **overrides: Options to use instead of the stored ones.

        Returns:
          The trained model.
        """
        return self.resolve_class().from_trained(self.options, trained, **overrides)

    # ── Files ─────────────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> Path:
        """Writes the configuration to `model_config.yaml`.

        Args:
          save_dir: Directory to write into. Created if missing.

        Returns:
          The path of the file.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / CONFIG_FILE
        with path.open("w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
        return path

    @classmethod
    def load(cls, load_dir: Path) -> ModelConfig:
        """Reads a configuration written by `save`.

        Args:
          load_dir: Directory holding `model_config.yaml`, or the file itself.

        Returns:
          The configuration.
        """
        path = load_dir if load_dir.suffix == ".yaml" else load_dir / CONFIG_FILE
        with path.open() as f:
            stored: dict[str, Any] = yaml.safe_load(f)
        return cls(**stored)

    @staticmethod
    def save_trained_matrices(model: Configurable, save_dir: Path) -> Path:
        """Writes a model's trained arrays to `trained_matrices.npz`.

        Args:
          model: The trained model.
          save_dir: Directory to write into. Created if missing.

        Returns:
          The path of the file.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / TRAINED_FILE
        np.savez_compressed(path, allow_pickle=False, **model.trained_arrays())
        return path

    @staticmethod
    def load_trained_matrices(save_dir: Path) -> dict[str, npt.NDArray[Any]]:
        """Reads arrays written by `save_trained_matrices`.

        Args:
          save_dir: Directory holding `trained_matrices.npz`.

        Returns:
          The arrays, by name.
        """
        with np.load(save_dir / TRAINED_FILE, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}


# ── Store ─────────────────────────────────────────────────────────────────────


def save_model_to_config(
    model: Configurable,
    save_dir: str | Path = MODEL_STORE,
    name: str | None = None,
    training_data_filename: str | None = None,
) -> tuple[ModelConfig, Path]:
    """Saves a trained model's configuration and arrays.

    Args:
      model: The trained model.
      save_dir: The store.
      name: Directory name within the store. `None` uses the configuration
        hash.
      training_data_filename: Name of the data the model was trained on.

    Returns:
      The configuration and the directory it was saved in.

    Raises:
      TypeError: If the model's class cannot be stored.
      RuntimeError: If the model is not trained.
    """
    config = ModelConfig.from_model(model, training_data_filename)
    path = Path(save_dir) / (config.to_hash() if name is None else name)
    ModelConfig.save_trained_matrices(model, path)
    config.save(path)
    return config, path


def load_model_from_config(
    q: str | None = None,
    config: ModelConfig | None = None,
    load_dir: str | Path = MODEL_STORE,
    **overrides: Any,
) -> Configurable | None:
    """Loads a saved model by name or by configuration.

    Args:
      q: Directory name within the store, such as a configuration hash. Takes
        precedence over `config`.
      config: Configuration whose hash names the directory.
      load_dir: The store.
      **overrides: Options to use instead of the stored ones, such as `device`.

    Returns:
      The trained model, or `None` if the store holds no such model.

    Raises:
      ValueError: If neither `q` nor `config` is given.
    """
    if q is None:
        if config is None:
            raise ValueError("pass q or config to load a model")
        q = config.to_hash()
    path = find_matching_config(load_dir, q)
    if path is None:
        return None
    return ModelConfig.load(path).to_model(ModelConfig.load_trained_matrices(path), **overrides)


def find_matching_config(search_dir: str | Path, query_hash: str, config_file: str = CONFIG_FILE) -> Path | None:
    """Finds a saved model by directory name.

    Args:
      search_dir: The store.
      query_hash: Directory name, such as a configuration hash.
      config_file: Name of the configuration file, `model_config.yaml` or
        `esn_config.yaml`.

    Returns:
      The directory, or `None` if it does not hold a complete saved model.
    """
    path = Path(search_dir) / query_hash
    if (path / config_file).is_file() and (path / TRAINED_FILE).is_file():
        return path
    return None


def auto_load_or_create[M: Configurable](
    model_class: type[M],
    data: Any,
    config_dir: str | Path = MODEL_STORE,
    auto_save: bool = True,
    force_create: bool = False,
    query_hash: str | None = None,
    training_data_filename: str | None = None,
    **options: Any,
) -> M:
    """Loads a saved model with matching options, or trains and saves a new one.

    Args:
      model_class: The model class.
      data: The training data.
      config_dir: The store.
      auto_save: Save a newly trained model.
      force_create: Train a new model even if one is saved.
      query_hash: Directory name to look up. `None` uses the hash of the
        configuration `options` and `training_data_filename` give.
      training_data_filename: Name of the training data.
      **options: Constructor options.

    Returns:
      The trained model.

    Raises:
      TypeError: If `model_class` cannot be stored, or the saved model is of
        another class.
    """
    if query_hash is None:
        config = ModelConfig.from_init_params(model_class, data, training_data_filename, **options)
        query_hash = config.to_hash()

    path = None if force_create else find_matching_config(config_dir, query_hash)
    if path is not None:
        model = ModelConfig.load(path).to_model(ModelConfig.load_trained_matrices(path))
        if not isinstance(model, model_class):
            raise TypeError(f"{path} holds a {type(model).__name__}, not a {model_class.__name__}")
        return model

    model = model_class.from_data(data, **options)
    if auto_save:
        save_model_to_config(model, config_dir, query_hash, training_data_filename)
    return model


def list_saved_configs(search_dir: str | Path = MODEL_STORE) -> list[dict[str, Any]]:
    """Lists the saved models in a store.

    Args:
      search_dir: The store.

    Returns:
      For each saved model, its `directory`, `name`, `model_class`, and
      `training_data_filename`. Empty if the store does not exist.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(search_dir).glob(f"*/{CONFIG_FILE}")):
        config = ModelConfig.load(path)
        rows.append(
            {
                "directory": path.parent,
                "name": path.parent.name,
                "model_class": config.model_class,
                "training_data_filename": config.training_data_filename,
            }
        )
    return rows


# ── Echo state networks ───────────────────────────────────────────────────────
#
# A. Nóvoa's `esn_config`, moved here unchanged apart from local replacements for
# `romda.utils` and the shared `find_matching_config`.

#: the trained-network store is a project-local cache under results/ (gitignored)
ESN_STORE = MODEL_STORE.parent / "esn_configs"
ESN_CONFIG_FILE = "esn_config.yaml"

INIT_KEYS = [  # fixed hyperparameter settings
    "N_units",
    "Win_type",
    "N_wash",
    "upsample",
    "bias_in",
    "bias_out",
    "norm_method",
    "connect",
    # state parameters
    "dt",
    "observed_idx",
    "update_reservoir",
    "update_state",
    "Wout_svd",
    "N_dim",
    # training parameters and hyperparameter optimization
    "t_train",
    "t_val",
    "t_test",
    "N_func_evals",
    "N_grid",
    "N_folds",
    "N_initial_rand",
    "N_split",
    "rho_range",
    "sigma_in_range",
    "tikh_range",
    "noise",
    "seed",
    "training_data_filename",
]

TRAINED_KEYS = [
    "norm",
    "shift",
    "rho",
    "sigma_in",
    "tikh",
    "Win",
    "W",
    "Wout",
    "validation_data",
    # parametric ESN: saved with the matrices, not hashed — the regime set
    # is already identified by `training_data_filename`
    "param_names",
    "param_values",
    "param_labels",
    "input_parameters",
]


@dataclass
class ESNConfig:
    """Configuration class for ESN_model to enable easy save/load without retraining."""

    # Model dimensions
    N_dim: int = 0
    observed_idx: list | None = None
    update_reservoir: bool = True
    update_state: bool = True
    Wout_svd: bool = False
    training_data_filename: str | None = None
    plot_training: bool = False

    # Time parameters.
    dt: float = 0.1
    t_train: float | None = None
    t_val: float | None = None
    t_test: float | None = None

    # ESN architecture parameters
    N_units: int = 50
    norm_method: str = "range"
    Win_type: str = "sparse"
    N_wash: int = 5
    upsample: int = 1
    bias_in: float = 0.1
    bias_out: float = 1.0
    connect: float = 3

    # Training parameters
    noise: float = 1e-2
    noise_type: str = "gauss"
    N_func_evals: int = 40
    N_grid: int = 5
    N_folds: int = 8
    N_split: int = 5
    N_initial_rand: int = 0

    # Hyperparameter optimization ranges
    rho_range: tuple[float, float] = (0.2, 0.8)
    sigma_in_range: tuple[float, float] = (-2, 2)
    tikh_range: tuple[float, ...] = (1e-6, 1e-9, 1e-12)
    hyperparameters_to_optimize: tuple[str, ...] = ("rho", "sigma_in", "tikh")

    # Additional metadata
    seed: int | None = 0
    config_hash: str | None = None  # Hash for matching configs

    # Parametric ESN fields — declared BEFORE the trained matrices: `asdict` order is
    # the order `ESN_model` applies them on load, and `input_parameters` must be set
    # before `norm`/`Win` so their (N_dim_in,)-shape validation sees the parameter rows
    param_names: tuple = ()
    param_values: tuple | None = None
    param_labels: tuple | None = None
    input_parameters: np.ndarray | None = None

    # Trained data keys (loaded separately)
    norm: np.ndarray | None = None
    shift: np.ndarray | None = None
    rho: float | None = None
    sigma_in: float | None = None
    tikh: float | None = None
    Win: np.ndarray | None = None
    W: np.ndarray | None = None
    Wout: np.ndarray | None = None
    validation_data: np.ndarray | None = None

    @staticmethod
    def _init_config_dict(case):
        return {key: getattr(case, key) for key in INIT_KEYS}

    def to_hash(self):
        """Compute a hash of the initial configuration parameters.

        Uses INITIAL parameters only (before any optimization), so configs match
        even if hyperparameter optimization produces slightly different results
        due to randomness.

        Returns
        -------
        str
            The first 16 hex characters of the SHA256 digest of the config.
        """

        hash_params = ESNConfig._init_config_dict(self)

        # Normalize types to ensure consistent hashing
        # Convert all numpy arrays and lists to consistent format
        for key in ["bias_in", "bias_out"]:
            if hash_params[key] is not None:
                if isinstance(hash_params[key], (float, int)):
                    hash_params[key] = [float(hash_params[key])]
                elif isinstance(hash_params[key], np.ndarray):
                    hash_params[key] = hash_params[key].tolist()

        # Normalize observed_idx to list
        if hash_params.get("observed_idx") is not None:
            if isinstance(hash_params["observed_idx"], np.ndarray):
                hash_params["observed_idx"] = hash_params["observed_idx"].tolist()
            elif not isinstance(hash_params["observed_idx"], list):
                hash_params["observed_idx"] = list(hash_params["observed_idx"])

        # Normalize tikh_range to tuple (canonical form)
        if hash_params.get("tikh_range") is not None:
            if isinstance(hash_params["tikh_range"], (list, np.ndarray)):
                hash_params["tikh_range"] = tuple(hash_params["tikh_range"])

        hash_params = convert_to_python_type(hash_params)

        # Convert to JSON string (sorted keys for consistency)
        hash_string = json.dumps(hash_params, sort_keys=True)

        # Compute SHA256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]

    @classmethod
    def from_esn_model(cls, esn_model):
        """Create a config from an existing `ESN_model` instance.

        Parameters
        ----------
        esn_model : ESN_model
            Instance to extract the configuration from.

        Returns
        -------
        ESNConfig
        """

        config_dict = cls._init_config_dict(esn_model)

        return cls(**config_dict)

    @classmethod
    def from_init_params(cls, **kwargs):
        """Create a config from initialization parameters (before training).

        Useful for checking if a matching config already exists.

        Parameters
        ----------
        **kwargs
            ESN initialization parameters.

        Returns
        -------
        ESNConfig
        """

        config_dict = {key: kwargs[key] for key in INIT_KEYS if key in kwargs}

        # If N_test, N_val, N_train are not provided, compute them from t_test, t_val, t_train and dt
        for key in ["train", "val", "test"]:
            if f"N_{key}" in kwargs.keys():
                config_dict[f"t_{key}"] = kwargs[f"N_{key}"] * kwargs["dt"]

        init_config = cls(**config_dict)
        if init_config.N_dim == 0:
            assert "data" in kwargs and kwargs["data"] is not None, (
                "N_dim not provided and data not available to infer it."
            )
            init_config.N_dim = kwargs["data"].shape[1]

        if init_config.observed_idx is None:
            init_config.observed_idx = list(range(init_config.N_dim))

        return init_config

    def to_esn_model(self, data=None, retrain: bool = False, **override_kwargs):
        """Create an `ESN_model` instance from this config.

        Parameters
        ----------
        data : np.ndarray, optional
            Training data; only needed if `retrain` is True.
        retrain : bool, optional
            If True, train a new model from `data` instead of loading a
            pre-trained one from the stored config. Default False.
        **override_kwargs
            Parameters to override from the config (only used if `retrain` is True).

        Returns
        -------
        ESN_model
            Loaded (pre-trained) or freshly trained model.
        """

        if not retrain:
            # Load pre-trained model
            if self.N_dim is None:
                raise ValueError("N_dim must be set to load a pre-trained model")

            config = asdict(self)
            config["y0"] = np.zeros((self.N_dim, 1))  # Dummy initial state
            # Parametric fields are absent/empty on non-parametric networks (and on
            # stores saved before they existed) — drop them so ESN_model's defaults apply
            for key in ("param_names", "param_values", "param_labels", "input_parameters"):
                v = config.get(key)
                if v is None or (np.ndim(v) == 0 and not np.any(v)) or (np.ndim(v) > 0 and len(v) == 0):
                    config.pop(key, None)
            return ESN_model(**config)
        else:
            # Create new model (requires training data)
            if data is None:
                raise ValueError("Training data required to create new ESN_model")

            init_config = asdict(self)
            init_config.update(override_kwargs)
            for key in TRAINED_KEYS:
                init_config.pop(key, None)  # Remove trained keys if present

            print("Creating and training new ESN_model...")
            return ESN_model(data=data, **init_config)

    def save(self, save_dir: Path):
        """Save configuration to YAML file."""
        filepath = Path(save_dir / "esn_config.yaml")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get initial parameters only
        config_dict = ESNConfig._init_config_dict(self)

        # Convert tuples to lists for YAML compatibility and  sort keys for consistent ordering in YAML file
        config_dict = convert_to_python_type(config_dict)
        config_dict = dict(sorted(config_dict.items()))  # type: ignore

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Print the last 3 folders and name of the save directory for confirmation
        print(f"Configuration saved to ...{'/'.join(list(save_dir.parts)[-3:])}/esn_config.yaml")

    @classmethod
    def load(cls, load_dir: Path, verbose=1):
        """Load configuration from YAML file."""
        # if yaml file is given, get its parent directory
        try:
            if load_dir.suffix == ".yaml":
                load_dir = load_dir.parent

            filepath = Path(load_dir / "esn_config.yaml")
            with open(filepath) as f:
                config_dict = yaml.safe_load(f)

            for key in config_dict.keys():
                if config_dict[key] == "none":
                    config_dict[key] = None

            if verbose:
                print(f"Configuration loaded from ...{'/'.join(list(load_dir.parts)[-3:])}/esn_config.yaml")
            return cls(**config_dict)

        except Exception as e:
            print(f"Error loading configuration from {load_dir}: {e}")
            raise e

    @staticmethod
    def save_trained_matrices(esn_model, save_dir: Path):
        """
        Save large matrices and other metadata defined after training the network.
        """
        data_to_save = {key: getattr(esn_model, key, False) for key in TRAINED_KEYS}

        np.savez_compressed(save_dir / "trained_matrices.npz", **data_to_save)
        print(f"Trained matrices saved to ...{'/'.join(list(save_dir.parts)[-3:])}/trained_matrices.npz")

    def update(self, update_dict: dict):
        """Update configuration parameters from a dictionary.

        Parameters
        ----------
        update_dict : dict
            Parameter name/value pairs to update (only existing attributes are set).
        """
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @staticmethod
    def load_trained_matrices(save_dir: Path):
        """Load saved matrices from a single ``.npz`` file.

        Parameters
        ----------
        save_dir : Path
            Directory containing the ``.npz`` file (e.g. ``'saved_configs/esn_abc123/'``).

        Returns
        -------
        dict
            Loaded matrices (`None` for every key if the file does not exist).
        """
        saved_file = Path(save_dir / "trained_matrices.npz")

        if not saved_file.exists():
            # Return dict with None values if file doesn't exist
            return {name: None for name in TRAINED_KEYS}

        trained_dict = {}
        with np.load(saved_file, allow_pickle=True) as data:
            for name in TRAINED_KEYS:
                item = data.get(name)
                if isinstance(item, np.ndarray) and item.dtype == object:
                    item = item.item()

                trained_dict[name] = item

        return trained_dict


# Convenience functions
def save_esn_model_to_config(esn_model, save_dir: str | Path = ESN_STORE, name: str | None = None):
    """Save an ESN model's configuration and trained matrices to disk.

    Parameters
    ----------
    esn_model : ESN_model
        Instance to save.
    save_dir : str or Path, optional
        Directory to save the configuration under.
    name : str, optional
        Subdirectory name for the saved files. Defaults to the config hash.

    Returns
    -------
    tuple
        ``(config, save_path)``.
    """

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    config = ESNConfig.from_esn_model(esn_model)

    if name is None:
        name = f"{config.to_hash()}"

    save_path = save_dir / name
    config.save(save_path)

    ESNConfig.save_trained_matrices(esn_model, save_path)

    return config, save_path


def load_esn_model_from_config(q: str | None = None, config: ESNConfig | None = None, load_dir: str | Path = ESN_STORE):
    """Load an `ESN_model` instance from a saved configuration.

    If `q` is given it takes priority; otherwise `config` is used to derive the
    query hash.

    Parameters
    ----------
    q : str, optional
        Query string (config hash) to match against saved configs.
    config : ESNConfig, optional
        Config instance to look up (via its hash) if `q` is not given.
    load_dir : str or Path, optional
        Directory to search for saved configs.

    Returns
    -------
    ESN_model or None
        The loaded model, or `None` if no matching config is found.
    """
    if q is not None:
        matching_path = find_matching_config(load_dir, q, ESN_CONFIG_FILE)

    elif config is not None:
        if isinstance(config, ESNConfig):
            q = config.to_hash()
        else:
            raise ValueError("Input config must be an instance of ESNConfig")

        matching_path = find_matching_config(load_dir, q, ESN_CONFIG_FILE)

    else:
        raise ValueError("Either query string q or config instance must be provided to load ESN_model")

    if not matching_path:
        print(f"No matching config {q} found in {load_dir}.")
        return None

    config = ESNConfig.load(matching_path)
    config.update(ESNConfig.load_trained_matrices(matching_path))

    return (
        config.to_esn_model()
    )  # Note: data is not needed to load a trained model since matrices are loaded separately


def auto_load_or_create_esn(
    config_dir: str | Path = ESN_STORE,
    auto_save: bool = True,
    force_create: bool = False,
    query_hash: str | None = None,
    data: np.ndarray | None = None,
    **kwargs,
):
    """Load a matching saved config, or create and train a new `ESN_model`.

    Parameters
    ----------
    config_dir : str or Path, optional
        Directory to search for/save configs.
    auto_save : bool, optional
        If True, save newly trained models. Default True.
    force_create : bool, optional
        If True, skip the search and always train a new model. Default False.
    query_hash : str, optional
        Hash to search for directly, skipping config-based hash computation.
    data : np.ndarray, optional
        Training data.
    **kwargs
        ESN initialization parameters.

    Returns
    -------
    ESN_model
        Loaded or newly trained model.
    """

    # If query_hash is provided, skip config creation and search directly for matching hash
    if query_hash is None:
        initial_params = kwargs.copy()
        initial_params["data"] = data
        query_config = ESNConfig.from_init_params(**initial_params)
        query_hash = query_config.to_hash()

    # print(f"Searching for config with hash: {query_hash}...")

    # Try to find matching config
    if force_create:
        print("Force creating new model (skipping search)...")
        matching_path = None
    else:
        # print(f"Searching for matching config... {config_dir} and hash {query_hash}")
        matching_path = find_matching_config(config_dir, query_hash, ESN_CONFIG_FILE)

    if matching_path:
        # Load existing model
        config = ESNConfig.load(matching_path)
        config.update(ESNConfig.load_trained_matrices(matching_path))
        return config.to_esn_model(data=data)
    else:
        # Train new model
        model = ESN_model(data=data, **kwargs)
        if auto_save:
            print(f"Saving new model to {config_dir}")
            save_esn_model_to_config(model, save_dir=config_dir, name=f"{query_hash}")
        return model


def list_saved_esn_configs(search_dir: str | Path, verbose=True):
    """List all saved ESN configs in a directory.

    Parameters
    ----------
    search_dir : str or Path
        Directory to search.
    verbose : bool, optional
        If True, print details about each config. Default True.

    Returns
    -------
    list of dict
        Summary info for each found config.
    """
    search_dir = Path(search_dir)
    configs = []

    if not search_dir.exists():
        search_dir = ESN_STORE.parent / search_dir
        if not search_dir.exists():
            print(f"Directory not found: {search_dir}")
            return configs

    for yaml_file in sorted(search_dir.glob("*/esn_config.yaml")):
        try:
            config = ESNConfig.load(yaml_file)

            config_info = {
                "directory": yaml_file.parent,
                "name": yaml_file.parent.name,
                "N_units": config.N_units,
                "dt": config.dt,
                "training_data_filename": config.training_data_filename,
            }

            configs.append(config_info)

            if verbose:
                print(f"\n{yaml_file.parent.name}:")
                print(f"  Training data: {config.training_data_filename}")
                print(f"  N_units: {config.N_units}")
                print(f"  N_dim: {config.N_dim}")
                print(f"  dt:  {config.dt}")
                print("  etc.: ...")

        except Exception as e:
            if verbose:
                print(f"Error reading {yaml_file}: {e}")

    return configs


def convert_to_python_type(obj, *, float_ndigits=12):
    """Convert numpy types to native Python types, with canonical float rounding."""
    if obj is None:
        return "none"

    if isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return round(float(obj), float_ndigits)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            c = complex(obj)
            return (round(c.real, float_ndigits), round(c.imag, float_ndigits))
        else:
            return obj.item()

    elif isinstance(obj, float):
        return round(obj, float_ndigits)

    elif isinstance(obj, np.ndarray):
        return [convert_to_python_type(x, float_ndigits=float_ndigits) for x in obj.tolist()]
    elif isinstance(obj, tuple):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v, float_ndigits=float_ndigits) for k, v in obj.items()}

    # if Path, change to sttring
    elif isinstance(obj, os.PathLike):
        return str(obj)

    return obj


# ── Helpers ───────────────────────────────────────────────────────────────────


def _class_path(model_class: type) -> str:
    """Returns the import path of a class, `module:QualifiedName`."""
    return f"{model_class.__module__}:{model_class.__qualname__}"


def _check_configurable(model_class: type[Configurable]) -> None:
    """Raises if a class cannot be stored.

    Raises:
      TypeError: If `model_class.configurable` is false.
    """
    if not model_class.configurable:
        raise TypeError(f"{model_class.__name__} cannot be stored; store its components instead")


def _canonical(value: Any) -> Any:
    """Normalizes a configuration value for hashing.

    Args:
      value: A configuration value.

    Returns:
      `value` with floats rounded and tuples as lists.
    """
    if isinstance(value, float):
        return round(value, _FLOAT_DIGITS)
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in cast(list[Any] | tuple[Any, ...], value)]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in cast(dict[Any, Any], value).items()}
    return value
