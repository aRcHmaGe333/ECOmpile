# Kernel: SID_REMOVE_SYSTEM_WIDE_KNOWN

KERNEL_ID: SID_REMOVE_SYSTEM_WIDE_KNOWN
STATUS: active
PLATFORM: windows
CONTEXT: powershell|cmd

## Intent Signature

Required tokens:
- sid
- remove
- system-wide or equivalent scope language
- user-provided known SID(s)

## Stop Condition

If known SID(s) are provided and request is system-wide permission removal, stop exploratory reasoning and emit primitive commands.

## Primitive

`icacls` recursive SID removal.

## Emit Template

```powershell
cmd /c 'icacls <ROOT> /remove *<SID1> /t /c /q'
cmd /c 'icacls <ROOT> /remove *<SID2> /t /c /q'
```

Default root for system-wide requests: `C:\`

## Forbid

Do not emit before primitive:
1. custom filesystem scanners,
2. orphan-detection scripts,
3. speculative remediation paths,
4. claims that no built-in method exists.

## Verify (Optional)

```powershell
cmd /c 'icacls <ROOT> | findstr /i "<SID>"'
```

## Source Case

- `cases/2026-03-04_unknown-contact-sid-removal/conversation.md`
