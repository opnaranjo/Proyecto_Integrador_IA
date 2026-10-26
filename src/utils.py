"""Funciones auxiliares: configuración, rutas, registro de eventos y persistencia."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def get_logger(name: str = "burnout", level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger con formato homogéneo para todo el proyecto.

    Args:
        name: nombre del logger.
        level: nivel mínimo de mensajes a registrar.

    Returns:
        Instancia de ``logging.Logger`` configurada una sola vez.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee el archivo YAML de configuración.

    Args:
        path: ruta al archivo; si es ``None`` se usa ``config.yaml`` de la raíz.

    Returns:
        Diccionario con la configuración.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Convierte una ruta relativa a la raíz del proyecto en ruta absoluta."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    """Fija las semillas de ``random`` y ``numpy`` para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: Any, path: str | Path) -> Path:
    """Guarda un objeto serializable como JSON con sangría y UTF-8."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
    return out


def load_json(path: str | Path) -> Any:
    """Carga un archivo JSON."""
    with open(resolve_path(path), encoding="utf-8") as fh:
        return json.load(fh)
