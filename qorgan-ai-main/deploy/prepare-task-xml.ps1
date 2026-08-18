<#
.SYNOPSIS
    Turn a Qorgan task template into a task definition schtasks will actually accept.

.DESCRIPTION
    Two jobs, both of which are load-bearing and neither of which is obvious.

    1. Substitute __QORGAN_ROOT__ and __QORGAN_USER__. The templates cannot know where
       the system will be installed or who will run it, and a task whose Command points
       at a directory literally called __QORGAN_ROOT__ is accepted by Task Scheduler
       without complaint and never runs.

    2. Write UTF-16. **`schtasks /create /xml` rejects a UTF-8 file outright** -- with
       and without a BOM -- as "ERROR: The task XML is malformed. (1,2)", an error that
       points at the XML declaration and tells you nothing about the encoding. This was
       measured on Windows 11 against these exact files, not assumed. The templates are
       kept as UTF-8 in the repository so they can be read, diffed and grepped like
       every other file here; this converts on the way out, and fixes the declaration to
       match the bytes.

    The task templates also declare version="1.4". Downgrading that to 1.2 makes
    schtasks reject them with "The task XML contains an unexpected node" -- also
    measured.

.NOTES
    Called by install-autostart.bat. Separate from it so that what it produces can be
    tested (tests/test_autostart.py) without registering a scheduled task on the machine
    running the tests.
#>
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$User
)

$ErrorActionPreference = 'Stop'

$xml = Get-Content -Raw -Encoding UTF8 $Source

# .Replace(), NOT -replace: -replace is a REGEX operator, and both of these values are
# Windows paths and account names full of backslashes. "C:\qorgan\deploy" as a regex
# replacement string eats its own separators, and the corruption is silent -- you get a
# registered task pointing somewhere that does not exist.
$xml = $xml.Replace('__QORGAN_ROOT__', $Root)
$xml = $xml.Replace('__QORGAN_USER__', $User)
$xml = $xml.Replace('encoding="UTF-8"', 'encoding="UTF-16"')

if ($xml -match '__QORGAN_') {
    throw "placeholder left unsubstituted in $Source - refusing to register a task that cannot run"
}

# UTF-16LE with a BOM: (bigEndian = $false, byteOrderMark = $true). WriteAllText rather
# than Set-Content/Out-File so the encoding is stated outright instead of depending on
# which PowerShell edition is running this.
$encoding = New-Object System.Text.UnicodeEncoding($false, $true)
[System.IO.File]::WriteAllText($Destination, $xml, $encoding)
