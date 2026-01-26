# -*- coding: utf-8 -*-
"""
Gera gráficos (PNG + PDF) a partir do summary_all_phases.json do projeto FL+CKKS.

Uso (Colab):
  %%bash
  python3 /content/drive/MyDrive/FL_CKKS_PROJETO/make_plots.py \
    --summary "/content/drive/MyDrive/FL_CKKS_PROJETO/outputs/summary_all_phases.json" \
    --outdir  "/content/drive/MyDrive/FL_CKKS_PROJETO/outputs/plots"
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(outdir: Path, name: str):
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def _barplot(phases, values, ylabel, title, outdir, fname, logy=False):
    plt.figure()
    xs = list(range(len(phases)))
    plt.bar(xs, values)
    plt.xticks(xs, phases, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    _save_fig(outdir, fname)


def _stacked_latency(phases, parts_dict, ylabel, title, outdir, fname):
    plt.figure()
    xs = list(range(len(phases)))
    bottom = [0.0] * len(phases)
    for label, vals in parts_dict.items():
        plt.bar(xs, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.xticks(xs, phases, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save_fig(outdir, fname)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_phase_records(summary):
    """
    Suporta:
      - lista: [ {...}, {...} ]
      - dict com 'phases': {'phases': [ ... ]}
      - dict com chaves de fase: {'Treinamento': {...}, 'Teste': {...}}
    Retorna lista de (phase_name, data_dict)
    """
    if isinstance(summary, list):
        out = []
        for i, item in enumerate(summary):
            if isinstance(item, dict):
                name = item.get("name") or item.get("phase") or item.get("phase_name") or f"phase_{i+1}"
                out.append((str(name), item))
        return out

    if isinstance(summary, dict) and isinstance(summary.get("phases"), list):
        out = []
        for i, item in enumerate(summary["phases"]):
            if isinstance(item, dict):
                name = item.get("name") or item.get("phase") or item.get("phase_name") or f"phase_{i+1}"
                out.append((str(name), item))
        return out

    if isinstance(summary, dict):
        out = []
        for k, v in summary.items():
            if isinstance(v, dict):
                out.append((str(k), v))
        return out

    return []


def _get_metric(d, candidates):
    for c in candidates:
        if c in d:
            return d.get(c)
    return None


def _metric_with_fallback(d, candidates):
    v = _get_metric(d, candidates)
    if v is not None:
        return v
    r = d.get("result")
    if isinstance(r, dict):
        return _get_metric(r, candidates)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Caminho do summary_all_phases.json")
    ap.add_argument("--outdir", required=True, help="Pasta de saída para os plots")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    summary = _read_json(summary_path)
    records = _extract_phase_records(summary)

    if not records:
        raise SystemExit(f"Não consegui interpretar o arquivo summary: {summary_path}")

    phase_names = []
    ctx_ms = []
    keygen_ms = []
    enc_ms = []
    agg_ms = []
    dec_ms = []
    payload_cipher_mb = []
    payload_total_mb = []
    mean_abs_error = []

    for name, d in records:
        phase_names.append(str(name))

        ctx = _safe_float(_metric_with_fallback(d, ["context_ms", "ckks_context_ms", "context_ckks_ms"]))
        key = _safe_float(_metric_with_fallback(d, ["keygen_ms", "ckks_keygen_ms"]))
        enc = _safe_float(_metric_with_fallback(d, ["encrypt_ms", "encrypt_clients_ms", "encrypt_clients_avg_ms"]))
        agg = _safe_float(_metric_with_fallback(d, ["aggregate_ms", "aggregate_avg_ms"]))
        dec = _safe_float(_metric_with_fallback(d, ["decrypt_ms", "decrypt_avg_ms"]))

        pc  = _safe_float(_metric_with_fallback(d, ["payload_cipher_mb", "payload_per_cipher_mb", "payload_cipher"]))
        pt  = _safe_float(_metric_with_fallback(d, ["payload_total_mb", "payload_total_avg_mb", "payload_total"]))

        err = _safe_float(_metric_with_fallback(d, ["mean_abs_error", "erro_medio_abs", "mean_absolute_error"]))

        # evita quebrar se faltar chave
        ctx_ms.append(ctx if ctx is not None else 0.0)
        keygen_ms.append(key if key is not None else 0.0)
        enc_ms.append(enc if enc is not None else 0.0)
        agg_ms.append(agg if agg is not None else 0.0)
        dec_ms.append(dec if dec is not None else 0.0)
        payload_cipher_mb.append(pc if pc is not None else 0.0)
        payload_total_mb.append(pt if pt is not None else 0.0)
        mean_abs_error.append(err if err is not None else 0.0)

    # Plots
    _barplot(phase_names, ctx_ms, "ms", "Latência: criação do contexto CKKS", outdir, "lat_context_ms")
    _barplot(phase_names, keygen_ms, "ms", "Latência: keygen CKKS", outdir, "lat_keygen_ms")
    _barplot(phase_names, enc_ms, "ms", "Latência: encrypt (clients)", outdir, "lat_encrypt_ms")
    _barplot(phase_names, agg_ms, "ms", "Latência: agregação", outdir, "lat_aggregate_ms")
    _barplot(phase_names, dec_ms, "ms", "Latência: decrypt", outdir, "lat_decrypt_ms")
    _barplot(phase_names, payload_cipher_mb, "MB", "Payload por ciphertext", outdir, "payload_cipher_mb")
    _barplot(phase_names, payload_total_mb, "MB", "Payload total por iteração", outdir, "payload_total_mb")
    _barplot(phase_names, mean_abs_error, "erro (abs)", "Erro médio absoluto (CKKS)", outdir, "mean_abs_error", logy=True)

    parts = {
        "Contexto": ctx_ms,
        "Keygen": keygen_ms,
        "Encrypt": enc_ms,
        "Aggregate": agg_ms,
        "Decrypt": dec_ms,
    }
    _stacked_latency(phase_names, parts, "ms", "Latência total por etapa (stacked)", outdir, "lat_total_stacked")

    readme = outdir / "README_plots.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"Plots gerados a partir de: {summary_path}\n")
        f.write("Arquivos: PNG (300dpi) e PDF.\n")
        f.write("Se alguma métrica ficou 0, a chave não existia no JSON.\n")

    print(f"✅ Plots gerados em: {outdir}")


if __name__ == "__main__":
    main()
