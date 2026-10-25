# Reproducibility Package v2.0 — Final Experimental and Security Campaign

Master's research:

**“Provendo Segurança a Modelos AI/ML com o Apoio da Criptografia Homomórfica e Aprendizado Federado”**

This document records the reproducibility freeze of the final experimental and security campaign. It is additive to, and does not replace or modify, the historical `reproducibility-v1.0` release.

## Release target

- Tag: `reproducibility-v2.0`
- Package: `THESIS_FINAL_REPRODUCIBILITY_PACKAGE_V2_VALIDATED.zip`
- SHA-256: `1761c97a1e5084cc167d846a20947fff099708f3db217e4c5863ae21d8b741bb`
- Step 1B validation: `PASS_STEP1B_V2_ROBUST_VALIDATION`

## Experimental scope — Gate 2

The final official campaign contains:

- 3 datasets:
  - PhysioNet Challenge 2012
  - Dahl Rats
  - CheXchoNet
- 3 scenarios:
  - Baseline
  - CKKS
  - Hybrid
- 3 training seeds:
  - 42
  - 123
  - 2026
- 30 federated rounds per experiment
- 5 clients per round

Validated scale:

- 27 completed experiments
- 810 federated rounds
- 4,050 local trainings
- 27/27 runs approved

Testing is performed after round 30. The protected scenarios preserve the scenario-specific update protection paths.

## AES-GCM integration

The final campaign integrates AES-GCM protection for global-model distribution from Server B to authorized clients.

- Server B encrypts the global model once per round.
- Five authorized clients decrypt locally before plaintext local training.
- A fresh unique 96-bit nonce is used for each encryption.
- The AES-GCM key is generated independently of ML seeds and is not persisted.
- AAD binds campaign/protocol metadata, dataset, scenario, training seed, round, model version, parameter count and dtype.
- AES-GCM protects confidentiality and integrity in transit.
- Application-level freshness/replay controls remain distinct from AES-GCM.
- AES-GCM does not replace CKKS or the Hybrid protection path.

The shared-key design and its compromise implications are explicit threat-model limitations.

## Security scope — Gate 3

The final security revalidation status is:

- Status: `PASS_FULL_SECURITY_REVALIDATION`
- Tested cases: 532
- Passed tested cases: 532
- `REVIEW_REQUIRED`: 0
- Execution failures: 0
- `NOT_TESTED_JUSTIFIED`: 6

The security campaign covers, among other controls:

- AES-GCM negative and freshness tests
- Gradient/update inversion stress tests
- Poisoning-to-GIA paired stress tests
- CKKS public-context probe
- CKKS and Hybrid aggregate-only decryption contracts
- Hybrid internal-exposure revalidation
- checkpoint integrity
- recovery contract
- communication fault/retry semantics
- logical Server A / Server B segregation and collusion analysis

The Poisoning-to-GIA experiment is a CGI-inspired targeted loss-amplification stress test and is not claimed as an exact reproduction of CGI or as a cryptographic break of CKKS.

## Explicit limitations

The campaign does not claim experimental validation of:

- independent physical hosts for Servers A and B
- root/host compromise resistance
- production TLS or production client-authentication deployment
- lateral-movement resistance
- distributed exactly-once semantics
- general Byzantine robustness
- general poisoning/backdoor robustness

Server A / Server B separation is logical in the evaluated environment. Non-collusion remains an architectural assumption. Violation of that assumption is analyzed as a trust-boundary failure, not as a break of the underlying cryptography.

Negative inversion results, where present, must not be interpreted as proof that inversion is impossible.

## Included final notebooks

The validated reproducibility package contains four final notebooks:

1. `GATE1_E2E_AES_GCM_V1_1_FIXED.ipynb`
2. `GATE2_FULL_E2E_AES_GCM_3SEEDS_30R.ipynb`
3. `GATE3_FULL_SECURITY_ALL_TESTS.ipynb`
4. `GATE3_FIX_CKKS_CONTEXT_AND_CLOSE.ipynb`

The package validation verified notebook presence and integrity together with the final Gate 2 and Gate 3 evidence.

## Reproducibility and provenance

The validated package includes a SHA-256 manifest for its files. The package was rebuilt and independently checked after the final notebooks were incorporated.

The final package SHA-256 is:

```text
1761c97a1e5084cc167d846a20947fff099708f3db217e4c5863ae21d8b741bb
```

The datasets are not redistributed unless their original licenses explicitly permit redistribution. Dataset acquisition and validation must follow their respective source terms and the project documentation.

## Versioning policy

`reproducibility-v1.0` is preserved as a historical reproducibility release.

`reproducibility-v2.0` represents the final integrated experimental/security campaign, including the AES-GCM global-model distribution protection and the closed Gate 3 security revalidation.

Results in this release are intended for the final dissertation evidence base. They must not silently replace values in earlier article versions that were frozen before this campaign.
