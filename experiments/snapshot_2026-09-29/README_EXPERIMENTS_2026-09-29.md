# Mestrado — Federated Learning + Cryptographic Protection

Snapshot experimental da pesquisa de mestrado.

## Campanha multi-seed
- Seeds: 42, 123 e 2026
- 3 datasets × 3 cenários × 3 seeds = 27 experimentos
- 30 rounds por experimento
- 5 clientes por round
- 810 rounds federados
- 4.050 treinamentos locais
- Cenários: Baseline, CKKS e Hybrid Transciphering

As Seeds 123 e 2026 possuem auditoria estrutural V3 aprovada. A Seed 42 pertence à campanha oficial previamente auditada.

## Segurança e tamanho do repositório
Datasets brutos, checkpoints, caches, chaves criptográficas, credenciais e tokens não são incluídos neste snapshot.

## Consolidação
A FASE23 consolida as três seeds com estatística descritiva (média e desvio-padrão amostral) e deltas pareados por seed. O arquivo `MISSING_METRICS_AUDIT.csv` preserva explicitamente campos não encontrados pelo extrator, sem preenchimento artificial.
