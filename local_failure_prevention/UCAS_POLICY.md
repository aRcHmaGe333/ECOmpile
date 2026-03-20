# UCAS Integration Policy

UCAS in this module means:

1. protect owner interests by default,
2. separate sensitive details from public-safe information,
3. avoid exposing internal strategy/context unless strictly required.

## Operational rules

1. Every analysis run generates two outputs:
- internal report (full details),
- public-safe report (redacted details).

2. Public-safe reports must remove or mask:
- direct contact addresses,
- phone numbers,
- local file paths,
- credentials/tokens/session markers.

3. Failure prevention actions use internal data only.

4. External sharing should use public-safe artifacts unless explicit override is approved.
