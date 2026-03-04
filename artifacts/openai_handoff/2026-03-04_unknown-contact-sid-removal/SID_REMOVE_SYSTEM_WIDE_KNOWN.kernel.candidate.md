# Kernel Candidate: SID_REMOVE_SYSTEM_WIDE_KNOWN

KERNEL_ID: SID_REMOVE_SYSTEM_WIDE_KNOWN
STATUS: candidate
PLATFORM: windows
CONTEXT: powershell|cmd

## Intent Signature
- sid
- remove
- system-wide

## Stop Condition
On intent match and validated primitive availability, stop exploratory branch generation and emit deterministic primitive template.

## Emit Template
- `cmd /c 'icacls <ROOT> /remove *<SID1> /t /c /q'`
- `cmd /c 'icacls <ROOT> /remove *<SID2> /t /c /q'`

## Forbid
- custom filesystem scanners
- orphan-detection scripts
- speculative remediation paths
- claims that no built-in method exists

## Source Case
- `cases/2026-03-04_unknown-contact-sid-removal/conversation.md`
