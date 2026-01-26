Plots gerados automaticamente para dissertação.

Fonte: bench_*.json em /content/drive/MyDrive/FL_CKKS_PROJETO/outputs (vector_size=None, clients=None)

Resumo (por fase):
	phase	lat_context_ms	lat_keygen_ms	lat_encrypt_ms	lat_aggregate_ms	lat_decrypt_ms	payload_cipher_mb	payload_total_mb	mean_abs_error
	oficial - chexchonet	325.378	11.686	36.4465	1.81732	1.61246	2.06221	2.06221	0.354591
	oficial chexchonet	37.2208	240.735	36.0688	2.13623	1.52047	1.5944	1.5944	4.87846e-09
	oficial chexchonet	58.9397	416.2	55.7272	3.088	2.17783	1.59417	1.59417	24.2381
	teste - dahl rats	269.256	10.2582	34.7126	1.72215	1.55081	2.06216	2.06216	0.33245
	teste dahl rats	38.106	253.143	37.2866	2.19401	1.4575	1.59413	1.59413	4.89643e-09
	teste dahl rats	60.9604	388.419	58.9042	3.03686	2.17389	1.5942	1.5942	24.0128
	treinamento - mimic-iii	305.766	10.9364	36.904	1.82678	1.72197	2.06219	2.06219	0.33245
	treinamento mimic iii	42.2023	278.554	40.3363	2.72664	1.74066	1.59404	1.59404	4.86373e-09
	treinamento mimic iii	84.7523	429.387	62.6195	3.32213	2.37925	1.59381	1.59381	24.1846