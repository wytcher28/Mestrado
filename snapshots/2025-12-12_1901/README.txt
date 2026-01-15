# Snapshot – FL com CKKS (Arquitetura A/B)

Este snapshot contém o estado final do experimento de Mestrado.

Conteúdo:
- Ambiente_Teste_FL.ipynb  -> notebook principal
- run_experimento_fl_ckks.py -> script de execução
- resultados/ -> CSVs de resultados
- graficos/ -> gráficos gerados
- requirements.txt -> dependências do ambiente

Base de dados:
- MIMIC-III Clinical Database (v1.4)
- Tarefa: predição de mortalidade hospitalar

Modelagem:
- Modelo: Regressão Logística
- Cenários: FL sem HE e FL com CKKS (Arquitetura A/B)

Para reprodução:
1) Montar Google Drive
2) pip install -r requirements.txt
3) Executar Ambiente_Teste_FL.ipynb
