#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ============================================================
# Helpers
# ============================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _norm(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _join(*parts: str) -> str:
    return _norm(os.path.join(*parts))


def _is_dir(p: str) -> bool:
    try:
        return os.path.isdir(p)
    except Exception:
        return False


def _is_file(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False


def _listdir_safe(p: str) -> List[str]:
    try:
        return os.listdir(p)
    except Exception:
        return []


def _walk_limited(root: str, max_depth: int = 4):
    """
    os.walk com profundidade limitada (para não varrer tudo sem necessidade).
    """
    root = _norm(root)
    base_sep = root.count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_sep
        if depth > max_depth:
            dirnames[:] = []
            continue
        yield dirpath, dirnames, filenames


def _find_first_existing_file(root: str, names: List[str]) -> Optional[str]:
    for n in names:
        candidate = _join(root, n)
        if _is_file(candidate):
            return candidate
    return None


# ============================================================
# Detectores
# ============================================================

def detect_sha256sums(dataset_root: str) -> Optional[str]:
    return _find_first_existing_file(dataset_root, ["SHA256SUMS.txt", "sha256sums.txt"])


def sha_mentions(sha_path: str, needle: str) -> bool:
    try:
        with open(sha_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().lower()
        return needle.lower() in txt
    except Exception:
        return False


def sha_mentions_metadata(sha_path: str) -> bool:
    return sha_mentions(sha_path, "metadata.csv")


def sha_suggests_images(sha_path: str) -> bool:
    """
    Heurística: se o SHA lista extensões de imagem, provavelmente é dataset de imagem.
    """
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
        if sha_mentions(sha_path, ext):
            return True
    return False


def detect_metadata_path(dataset_root: str) -> Optional[str]:
    """
    Procura por metadata/labels/annotations etc.
    1) No root
    2) Recursivo até depth 4
    """
    dataset_root = _norm(dataset_root)

    direct_candidates = [
        "metadata.csv", "Metadata.csv", "METADATA.csv",
        "labels.csv", "label.csv", "annotations.csv", "targets.csv",
    ]
    direct = _find_first_existing_file(dataset_root, direct_candidates)
    if direct:
        return direct

    wanted_suffixes = ("metadata.csv", "labels.csv", "annotations.csv", "targets.csv")
    for dirpath, _, filenames in _walk_limited(dataset_root, max_depth=4):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(wanted_suffixes) or low.endswith("_metadata.csv") or low.endswith("_labels.csv"):
                return _join(dirpath, fn)

    return None


def detect_images_dir(dataset_root: str) -> Optional[str]:
    """
    1) Se existir dataset_root/images, retorna.
    2) Senão, procura pasta com mais imagens (até depth 3).
    Só retorna se encontrar quantidade razoável (>= 50) para evitar falsos positivos.
    """
    dataset_root = _norm(dataset_root)

    direct = _join(dataset_root, "images")
    if _is_dir(direct):
        return direct

    best_dir = None
    best_count = 0

    for dirpath, _, filenames in _walk_limited(dataset_root, max_depth=3):
        local = sum(1 for fn in filenames if fn.lower().endswith(IMAGE_EXTS))
        if local > best_count:
            best_count = local
            best_dir = dirpath

    if best_dir and best_count >= 50:
        return best_dir

    return None


def count_images(root: str, max_depth: int = 4) -> int:
    count = 0
    for _, _, filenames in _walk_limited(root, max_depth=max_depth):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                count += 1
    return count


def top_level_subfolders(dataset_root: str, limit: int = 20) -> List[str]:
    subs = []
    for name in _listdir_safe(dataset_root):
        p = _join(dataset_root, name)
        if _is_dir(p):
            subs.append(name)
    return sorted(subs)[:limit]


def prefer_extracted_root(phase_root: str) -> str:
    """
    Se existir phase_root/extracted com subpasta(s),
    escolhe a subpasta mais provável (1 só => retorna ela;
    várias => heurística: tem SHA e mais itens).
    """
    phase_root = _norm(phase_root)
    extracted = _join(phase_root, "extracted")
    if not _is_dir(extracted):
        return phase_root

    subdirs = [d for d in _listdir_safe(extracted) if _is_dir(_join(extracted, d))]
    if not subdirs:
        return phase_root

    if len(subdirs) == 1:
        return _join(extracted, subdirs[0])

    scored: List[Tuple[int, int, str]] = []
    for d in subdirs:
        root = _join(extracted, d)
        has_sha = 1 if detect_sha256sums(root) else 0
        n_items = len(_listdir_safe(root))
        scored.append((has_sha, n_items, root))

    scored.sort(reverse=True)
    return scored[0][2]


# ============================================================
# Fase
# ============================================================

@dataclass
class PhaseConfig:
    name: str
    root: str
    images_dir: Optional[str]
    mode: str
    metadata_path: Optional[str]


def build_phase(name: str, phase_root: str) -> PhaseConfig:
    root = prefer_extracted_root(phase_root)
    sha = detect_sha256sums(root)

    metadata_path = detect_metadata_path(root)
    images_dir = detect_images_dir(root)

    mode = "supervised" if metadata_path else "unlabeled"

    return PhaseConfig(
        name=name,
        root=root,
        images_dir=images_dir,
        mode=mode,
        metadata_path=metadata_path,
    )


def default_phases(base_data: str) -> List[PhaseConfig]:
    base = _norm(base_data)
    mapping = [
        ("Treinamento – MIMIC-III", _join(base, "Treino")),
        ("Teste – Dahl Rats", _join(base, "Teste")),
        ("Oficial – CheXchoNet", _join(base, "Oficial")),
    ]
    return [build_phase(name, root) for name, root in mapping]


# ============================================================
# Relatório (no formato do seu print)
# ============================================================

def print_phase_report(phase: PhaseConfig) -> None:
    sha = detect_sha256sums(phase.root)

    # define se parece dataset de imagem:
    likely_image_dataset = False
    if _is_dir(_join(phase.root, "images")):
        likely_image_dataset = True
    elif sha and sha_suggests_images(sha):
        likely_image_dataset = True

    print("=" * 60)
    print(f"Fase experimental: {phase.name}")
    print(f"Root dataset : {phase.root}")
    print(f"Modo         : {phase.mode}")
    print(f"Metadata     : {'ENCONTRADA' if phase.metadata_path else 'NÃO encontrada'}")

    if phase.metadata_path:
        print("→ Métricas supervisionadas HABILITADAS")
        print(f"  metadata_path: {phase.metadata_path}")
    else:
        print("→ Métricas supervisionadas DESABILITADAS")
        print("  (somente latência, custo, overhead, escalabilidade)")

    # Images dir
    if phase.images_dir:
        print(f"Images dir   : {phase.images_dir}")
    else:
        # só avisa se fizer sentido (dataset imagem)
        if likely_image_dataset:
            print("  ⚠️  Aviso: images_dir não detectado (pasta de imagens não encontrada).")

    # SHA256SUMS diagnósticos
    if sha:
        print("  ℹ️  Nota: existe SHA256SUMS no dataset (lista de arquivos),")
        print("     mas isso não garante que os arquivos listados foram extraídos.")
        print(f"     SHA256SUMS: {sha}")

        if (not phase.metadata_path) and sha_mentions_metadata(sha):
            print("  ⚠️  Atenção: 'metadata.csv' aparece no SHA256SUMS,")
            print("     mas o arquivo NÃO foi encontrado no disco (não extraído/copiado).")

    # subpastas e contagem de imagens
    subs = top_level_subfolders(phase.root)
    if subs:
        print(f"  📁 Subpastas (top): {', '.join(subs)}")

    if phase.images_dir:
        n_imgs = count_images(phase.images_dir, max_depth=4)
        print(f"  🖼️  Imagens detectadas (até depth 4): {n_imgs}")

    print("Status       : fase validada e pronta para execução")
    print("=" * 60)


def phases_as_dict(phases: List[PhaseConfig]) -> List[Dict]:
    return [asdict(p) for p in phases]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detecta e valida fases experimentais.")
    ap.add_argument("--base-data", required=True, help='Ex: "/content/drive/MyDrive/Base de Dados"')
    ap.add_argument("--quiet", action="store_true", help="Não imprimir relatórios.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base_data = _norm(args.base_data)

    if not _is_dir(base_data):
        print(f"ERRO: base-data não existe ou não é diretório: {base_data}")
        return 2

    phases = default_phases(base_data)

    if not args.quiet:
        for p in phases:
            print_phase_report(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
