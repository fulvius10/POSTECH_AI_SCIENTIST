# Setup do ambiente dos hands-on — Curso IA Generativa (aulas 01 a 06)
# Uso:  powershell -ExecutionPolicy Bypass -File setup\instalar.ps1
#
# Cria o venv em C:\git\aula\.venv, instala torch (CUDA se houver GPU NVIDIA)
# e as demais dependências do requirements.txt.

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot   # C:\git\aula
$venv = Join-Path $raiz ".venv"

Write-Host "==> Criando venv em $venv (Python padrao da maquina)..."
python -m venv $venv

$pip = Join-Path $venv "Scripts\pip.exe"
$py  = Join-Path $venv "Scripts\python.exe"

& $pip install --upgrade pip

# torch: tenta o build CUDA 12.6 (GPU NVIDIA); se nao houver GPU, use o build CPU
$temGpu = $false
try { nvidia-smi | Out-Null; $temGpu = $true } catch {}
if ($temGpu) {
    Write-Host "==> GPU NVIDIA detectada - instalando torch com CUDA 12.6..."
    & $pip install torch --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "==> Sem GPU NVIDIA - instalando torch CPU..."
    & $pip install torch
}

Write-Host "==> Instalando demais dependencias..."
& $pip install -r (Join-Path $raiz "requirements.txt")

Write-Host "==> Registrando o kernel Jupyter do venv (caminho absoluto)..."
& $py -m ipykernel install --prefix $venv --name python3 --display-name "Python (curso IA Generativa)"

Write-Host "==> Rodando smoke test..."
& $py (Join-Path $PSScriptRoot "testar_ambiente.py")

Write-Host ""
Write-Host "Pronto. Para ativar o ambiente neste terminal:"
Write-Host "  $venv\Scripts\Activate.ps1"
Write-Host "Proximo passo (fazer com antecedencia, baixa ~9 GB):"
Write-Host "  python setup\baixar_modelos.py"
