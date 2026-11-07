$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\anbee\projects\beellama-v050-sinq'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
cmd /c "`"$vcvars`" x64 > nul && set" | ForEach-Object { if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
cmake --build build-sinq-clean --target llama-perplexity llama-sinq-model llama-sinq-native llama-quantize llama-completion test-sinq-candidates test-sinq-bridge-config test-sinq-threading test-perplexity-full test-perplexity-plumbing test-qk-sinq-numeric test-qk-sinq-model test-qk-sinq-graph test-qk-sinq-backend --parallel 16
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
