# Mestrado — FL + CKKS (TenSEAL)

Este repositório contém código e snapshots de experimentos de **Federated Learning (FL)** com **Criptografia Homomórfica aproximada (CKKS)** usando **TenSEAL**, incluindo scripts de execução, resultados e gráficos.

## Estrutura do repositório

- `FULL_PROJECT.py`  
  Arquivo único gerado por concatenação dos scripts `.py` do projeto (útil para leitura/cópia rápida).

- `snapshots/2025-12-12_1901/`  
  Snapshot completo do experimento (código, requisitos, notebook, resultados e gráficos).
  - `run_experimento_fl_ckks.py`
  - `requirements.txt`
  - `Ambiente_Teste_FL.ipynb`
  - `resultados/`
  - `graficos/`

## Como executar (snapshot)

1. Crie um ambiente (Python 3.10+ recomendado) e instale dependências:

```bash
pip install -r snapshots/2025-12-12_1901/requirements.txt
