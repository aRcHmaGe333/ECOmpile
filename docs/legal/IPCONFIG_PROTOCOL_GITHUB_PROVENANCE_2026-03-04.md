# IPConfig Protocol: GitHub Provenance Requirement

Protocol ID: IPCONFIG-PROT-GITHUB-PROVENANCE-2026-03-04
Status: Active
Effective date: 2026-03-04

## Rule

IPConfig/IPClaim evidence is operational when anchored in GitHub provenance:

1. committed history,
2. pushed remote state,
3. timestamp workflow configuration,
4. timestamp artifact path availability.

## Minimum Validity Conditions

1. Changes committed in git.
2. Commit pushed to `origin`.
3. Timestamp automation present at `.github/workflows/timestamp.yml`.
4. Timestamp artifact path exists at `.timestamps/`.

## Linked Records

1. `docs/legal/IPCONFIG_PROTOCOL_DATE_CONTINUITY_2026-03-04.md`
2. `docs/legal/IP_PROVENANCE_REGISTER.md`
3. `docs/legal/IPCONFIG_PROOF.md`
4. `docs/legal/IPCLAIM_APPLICATION_LOG_2026-03-04.md`
