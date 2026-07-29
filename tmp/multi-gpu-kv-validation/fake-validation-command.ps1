param([string] $Mode, [Parameter(ValueFromRemainingArguments)] [string[]] $Rest)
if ($Mode -eq 'semantic') {
    [Console]::Error.WriteLine('critical error: injected harness semantic failure')
    exit 0
}
if ($Mode -eq 'timeout') {
    Start-Sleep -Seconds 30
}
[Console]::Out.WriteLine('validation output: OK')
