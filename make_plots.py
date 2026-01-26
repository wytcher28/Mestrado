#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Util
# -----------------------------
def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_text(p: Path, s: str):
    with p.open("w", encoding="utf-8") as f:
        f.write(s)

def _autolabel(ax, rects, fmt="{:.2f}", ypad=3):
    # coloca valores em cima das barras
    for r in rects:
        h = r.get_height()
        if h is None:
            continue
        if math.isnan(h):
            continue
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, ypad),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

def _savefig(outdir: Path, name: str):
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    plt.close()

def _pretty_phase_name(s: str):
    # tenta deixar mais legível
    s = s.replace("__", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# Leitura de bench_*.json
# -----------------------------
def load_bench_files(outputs_dir: Path, vector_size: int | None, clients: int | None):
    bench_files = sorted(outputs_dir.glob("bench_*.json"))
    if not bench_files:
        raise FileNotFoundError(f"Nenhum bench_*.json encontrado em: {outputs_dir}")

    rows = []
    for bf in bench_files:
        data = _read_json(bf)

        # tenta inferir fase pelo filename: bench_oficial__chexchonet_v4096_c5.json
        stem = bf.stem
        m = re.search(r"^bench_(.+)_v(\d+)_c(\d+)$", stem)
        phase_name = stem
        v = None
        c = None
        if m:
            phase_name = m.group(1)
            v = int(m.group(2))
            c = int(m.group(3))

        # filtra, se pedido
        if vector_size is not None and v is not None and v != vector_size:
            continue
        if clients is not None and c is not None and c != clients:
            continue

        # normaliza chaves (os seus scripts já imprimem coisas tipo "lat_context_ms")
        row = {
            "phase": _pretty_phase_name(phase_name),
            "file": str(bf),
            "vector_size": v,
            "clients": c,
            "lat_context_ms": _safe_float(data.get("lat_context_ms")),
            "lat_keygen_ms": _safe_float(data.get("lat_keygen_ms")),
            "lat_encrypt_ms": _safe_float(data.get("lat_encrypt_ms")),
            "lat_aggregate_ms": _safe_float(data.get("lat_aggregate_ms")),
            "lat_decrypt_ms": _safe_float(data.get("lat_decrypt_ms")),
            "payload_cipher_mb": _safe_float(data.get("payload_cipher_mb")),
            "payload_total_mb": _safe_float(data.get("payload_total_mb")),
            "mean_abs_error": _safe_float(data.get("mean_abs_error")),
        }

        # fallback: alguns JSON usam t_* ao invés de lat_*
        if row["lat_context_ms"] is None:   row["lat_context_ms"]   = _safe_float(data.get("t_context_ms"))
        if row["lat_keygen_ms"] is None:    row["lat_keygen_ms"]    = _safe_float(data.get("t_keygen_ms"))
        if row["lat_encrypt_ms"] is None:   row["lat_encrypt_ms"]   = _safe_float(data.get("t_encrypt_ms"))
        if row["lat_aggregate_ms"] is None: row["lat_aggregate_ms"] = _safe_float(data.get("t_aggregate_ms"))
        if row["lat_decrypt_ms"] is None:   row["lat_decrypt_ms"]   = _safe_float(data.get("t_decrypt_ms"))

        # fallback: payload em bytes
        if row["payload_cipher_mb"] is None and data.get("payload_cipher_bytes") is not None:
            row["payload_cipher_mb"] = _safe_float(data.get("payload_cipher_bytes")) / (1024*1024)
        if row["payload_total_mb"] is None and data.get("payload_total_bytes") is not None:
            row["payload_total_mb"] = _safe_float(data.get("payload_total_bytes")) / (1024*1024)

        # fallback: nome de clientes
        if row["clients"] is None:
            if data.get("num_clients") is not None:
                row["clients"] = int(data.get("num_clients"))

        # alguns logs podem usar nomes diferentes
        # tenta mapear se necessário
        if row["lat_encrypt_ms"] is None:
            row["lat_encrypt_ms"] = _safe_float(data.get("lat_encrypt_clients_ms"))
        if row["payload_total_mb"] is None:
            row["payload_total_mb"] = _safe_float(data.get("payload_total_mb_avg"))
        if row["payload_cipher_mb"] is None:
            row["payload_cipher_mb"] = _safe_float(data.get("payload_cipher_mb_avg"))

        rows.append(row)

    if not rows:
        raise ValueError(
            f"Encontrei bench_*.json, mas nenhum bateu com filtros "
            f"(vector_size={vector_size}, clients={clients})."
        )

    return rows

# -----------------------------
# Leitura de summary_all_phases.json
# (suporta list ou dict)
# -----------------------------
def load_summary(summary_path: Path):
    data = _read_json(summary_path)
    rows = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # pode ser {"phases": [...]} ou {"results": [...]}
        if "phases" in data and isinstance(data["phases"], list):
            items = data["phases"]
        elif "results" in data and isinstance(data["results"], list):
            items = data["results"]
        else:
            # tenta tratar dict {phase_name: {...}}
            items = []
            for k, v in data.items():
                if isinstance(v, dict):
                    v = dict(v)
                    v.setdefault("name", k)
                    items.append(v)
    else:
        raise ValueError("Formato de summary inválido (nem list nem dict).")

    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("phase") or "phase"
        # tenta achar métricas
        row = {
            "phase": _pretty_phase_name(str(name)),
            "file": str(summary_path),
            "vector_size": _safe_float(it.get("vector_size")),
            "clients": _safe_float(it.get("clients")),
            "lat_context_ms": _safe_float(it.get("lat_context_ms")),
            "lat_keygen_ms": _safe_float(it.get("lat_keygen_ms")),
            "lat_encrypt_ms": _safe_float(it.get("lat_encrypt_ms")),
            "lat_aggregate_ms": _safe_float(it.get("lat_aggregate_ms")),
            "lat_decrypt_ms": _safe_float(it.get("lat_decrypt_ms")),
            "payload_cipher_mb": _safe_float(it.get("payload_cipher_mb")),
            "payload_total_mb": _safe_float(it.get("payload_total_mb")),
            "mean_abs_error": _safe_float(it.get("mean_abs_error")),
        }
        rows.append(row)

    if not rows:
        raise ValueError("Summary lido, mas não consegui extrair nenhuma linha de resultados.")
    return rows

# -----------------------------
# Plotters
# -----------------------------
def bar_plot(rows, key, title, ylabel, outdir: Path, fname, fmt="{:.2f}"):
    phases = [r["phase"] for r in rows]
    vals = [r.get(key) if r.get(key) is not None else float("nan") for r in rows]

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()
    rects = ax.bar(phases, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(phases, rotation=15, ha="right")
    _autolabel(ax, rects, fmt=fmt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _savefig(outdir, fname)

def stacked_latency(rows, outdir: Path, fname="lat_total_stacked"):
    phases = [r["phase"] for r in rows]
    ctx = [r.get("lat_context_ms") or 0 for r in rows]
    keyg = [r.get("lat_keygen_ms") or 0 for r in rows]
    enc = [r.get("lat_encrypt_ms") or 0 for r in rows]
    agg = [r.get("lat_aggregate_ms") or 0 for r in rows]
    dec = [r.get("lat_decrypt_ms") or 0 for r in rows]

    plt.figure(figsize=(10, 5.2))
    ax = plt.gca()
    b1 = ax.bar(phases, ctx, label="Context")
    b2 = ax.bar(phases, keyg, bottom=ctx, label="Keygen")
    b3 = ax.bar(phases, enc, bottom=[ctx[i]+keyg[i] for i in range(len(rows))], label="Encrypt")
    b4 = ax.bar(phases, agg, bottom=[ctx[i]+keyg[i]+enc[i] for i in range(len(rows))], label="Aggregate")
    b5 = ax.bar(phases, dec, bottom=[ctx[i]+keyg[i]+enc[i]+agg[i] for i in range(len(rows))], label="Decrypt")

    totals = [ctx[i]+keyg[i]+enc[i]+agg[i]+dec[i] for i in range(len(rows))]
    for i, t in enumerate(totals):
        ax.text(i, t + max(totals)*0.01 if max(totals)>0 else t+1, f"{t:.1f} ms",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Latência total por iteração (stacked)")
    ax.set_ylabel("ms")
    ax.set_xticklabels(phases, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncols=3, fontsize=9)
    _savefig(outdir, fname)

def write_readme(outdir: Path, rows, source_desc: str):
    lines = []
    lines.append("Plots gerados automaticamente para dissertação.")
    lines.append("")
    lines.append(f"Fonte: {source_desc}")
    lines.append("")
    # tabela simples
    hdr = ["phase","lat_context_ms","lat_keygen_ms","lat_encrypt_ms","lat_aggregate_ms","lat_decrypt_ms","payload_cipher_mb","payload_total_mb","mean_abs_error"]
    lines.append("Resumo (por fase):")
    lines.append("\t" + "\t".join(hdr))
    for r in rows:
        vals = [r.get(h) for h in hdr]
        fmtvals = []
        for v in vals:
            if isinstance(v, (int,float)) and v is not None:
                fmtvals.append(f"{v:.6g}")
            else:
                fmtvals.append(str(v))
        lines.append("\t" + "\t".join(fmtvals))
    _write_text(outdir / "README_plots.txt", "\n".join(lines))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Gera plots (PNG+PDF) a partir dos resultados do projeto (bench_*.json e/ou summary_all_phases.json)."
    )
    ap.add_argument("--outdir", required=True, help="Pasta destino para salvar os plots.")
    ap.add_argument("--outputs-dir", default=None, help="Pasta outputs/ contendo bench_*.json (preferencial).")
    ap.add_argument("--summary", default=None, help="Caminho do summary_all_phases.json (alternativo).")
    ap.add_argument("--vector-size", type=int, default=None, help="Filtra por vector-size (ex: 4096).")
    ap.add_argument("--clients", type=int, default=None, help="Filtra por número de clientes (ex: 5).")

    args = ap.parse_args()
    outdir = _mkdir(Path(args.outdir))

    rows = None
    source_desc = None

    if args.outputs_dir:
        rows = load_bench_files(Path(args.outputs_dir), args.vector_size, args.clients)
        source_desc = f"bench_*.json em {args.outputs_dir} (vector_size={args.vector_size}, clients={args.clients})"
    elif args.summary:
        rows = load_summary(Path(args.summary))
        source_desc = f"summary: {args.summary}"
    else:
        raise SystemExit("Erro: use --outputs-dir OU --summary")

    # ordena por fase para ficar estável
    rows = sorted(rows, key=lambda r: r["phase"])

    # gráficos individuais
    bar_plot(rows, "lat_context_ms",   "Latência: criação de contexto CKKS", "ms", outdir, "lat_context_ms", fmt="{:.1f}")
    bar_plot(rows, "lat_keygen_ms",    "Latência: key generation",          "ms", outdir, "lat_keygen_ms", fmt="{:.1f}")
    bar_plot(rows, "lat_encrypt_ms",   "Latência: encrypt (clients)",       "ms", outdir, "lat_encrypt_ms", fmt="{:.1f}")
    bar_plot(rows, "lat_aggregate_ms", "Latência: aggregate",               "ms", outdir, "lat_aggregate_ms", fmt="{:.2f}")
    bar_plot(rows, "lat_decrypt_ms",   "Latência: decrypt",                 "ms", outdir, "lat_decrypt_ms", fmt="{:.2f}")

    stacked_latency(rows, outdir, "lat_total_stacked")

    bar_plot(rows, "payload_cipher_mb", "Payload: tamanho do ciphertext", "MB", outdir, "payload_cipher_mb", fmt="{:.2f}")
    bar_plot(rows, "payload_total_mb",  "Payload: tráfego total por iteração", "MB", outdir, "payload_total_mb", fmt="{:.2f}")

    # erro pode ser muito grande/pequeno — usa notação científica
    bar_plot(rows, "mean_abs_error", "Erro médio absoluto (CKKS)", "erro", outdir, "mean_abs_error", fmt="{:.3g}")

    write_readme(outdir, rows, source_desc)
    print(f"Plots gerados em: {outdir}")

if __name__ == "__main__":
    main()
