# Campanha oficial da dissertação — V2

Campanha experimental congelada: THESIS_OFFICIAL_CAMPAIGN_V2_20260901.

## Matriz experimental

| Item | Configuração |
|---|---|
| Datasets | PhysioNet/Computing in Cardiology Challenge 2012; Dahl Rats; CheXchoNet |
| Cenários | Baseline; CKKS; Hybrid |
| Rodadas federadas | 30 |
| Clientes | 5 |
| Seed | 42 |
| Particionamento | não-IID congelado, com unidade antivasamento própria de cada dataset |
| Protocolo | phase18_official_v1_common_local_sgd_hybrid_ts0125 |
| Escala de transporte híbrido | 0.125 |

O gate final registra PHASE18_APPROVED=true, nove experimentos concluídos e resultados
autorizados para uso na dissertação.

## Estrutura

- notebooks/: versões oficiais selecionadas das Fases 16–18;
- results/: comparação consolidada, gate mestre e evidências;
- manifests/: manifesto e hashes SHA-256.

Execute os notebooks pela numeração das fases. A Fase 18 é a campanha oficial.
Datasets, dados clínicos, chaves, tokens e checkpoints temporários não são publicados.
