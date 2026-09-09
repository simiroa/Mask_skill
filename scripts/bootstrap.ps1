<#
맨몸 윈도우용 진입점. 파이썬이 없어도 돈다 — check_env.py 는 파이썬이 있어야 하므로
그 앞단을 여기서 맡는다.

★기본은 진단만 한다. 아무것도 바꾸지 않는다.
  실제 설치는 -Yes 를 줘야 한다. 용량·경로를 먼저 보여주고 받는 허락이다.

  .\bootstrap.ps1                      # 진단 + 설치 계획(용량·경로) 출력
  .\bootstrap.ps1 -Yes                 # 계획대로 설치
  .\bootstrap.ps1 -Yes -FetchModels    # 모델까지 미리 받음(첫 실행 대기 없음)
#>
[CmdletBinding()]
param(
    [string]$Env = "",                       # 가상환경 경로. 비우면 <스킬>\.venv-seg
    [switch]$Yes,                            # 실제로 설치한다
    [switch]$FetchModels,                    # OneFormer/CascadePSP 를 미리 받는다
    [ValidateSet("auto", "cu126", "cu128", "cpu")]
    [string]$Cuda = "auto"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Env) { $Env = Join-Path $root ".venv-seg" }

# 설치 용량. 다운로드는 압축 기준, 디스크는 풀린 기준.
# ★실측(RTX 3080 Ti / py3.12 / cu126): venv 5.1GB(torch 3.9GB) + 모델 1.9GB.
$PLAN = @(
    @{ 항목 = "Python 3.12 (winget)"; 다운 = 25MB;   디스크 = 120MB },
    @{ 항목 = "torch + torchvision";  다운 = 2.5GB;  디스크 = 3.9GB },
    @{ 항목 = "transformers 외 4종";  다운 = 150MB;  디스크 = 1.2GB },
    @{ 항목 = "OneFormer Swin-L";     다운 = 1.64GB; 디스크 = 1.64GB },
    @{ 항목 = "CascadePSP";           다운 = 271MB;  디스크 = 271MB }
)

function FmtB($n) {
    foreach ($u in "B", "KB", "MB", "GB", "TB") {
        if ($n -lt 1024 -or $u -eq "TB") {
            if ($u -eq "B") { return "{0:N0}{1}" -f $n, $u }
            return "{0:N1}{1}" -f $n, $u
        }
        $n = $n / 1024
    }
}

function Have($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

# ---------------------------------------------------------------- 진단

Write-Host "[환경 진단]"
Write-Host ("  OS       " + (Get-CimInstance Win32_OperatingSystem).Caption)
Write-Host ("  스킬     " + $root)

$winget = Have "winget"
Write-Host ("  winget   " + $(if ($winget) { (winget --version) } else { "없음 — Python 자동 설치 불가" }))

# GPU. nvidia-smi 는 드라이버와 함께 깔리므로 파이썬 없이도 읽힌다.
$driver = ""; $vram = 0; $gpu = ""; $cudaCap = ""
if (Have "nvidia-smi") {
    $q = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) -join ""
    if ($q -match '^(.+?),\s*([\d.]+),\s*(\d+)\s*MiB') {
        $gpu = $matches[1]; $driver = $matches[2]; $vram = [int]$matches[3]
    }
    # ★헤더 문구가 드라이버 세대마다 다르다. 610 대는 "CUDA UMD Version",
    #   그 이전은 "CUDA Version". 하나만 찾으면 조용히 실패한다.
    $hdr = (nvidia-smi) -join "`n"
    if ($hdr -match 'CUDA(?:\s+UMD)?\s+Version:\s*([\d.]+)') { $cudaCap = $matches[1] }
}
if ($gpu) {
    Write-Host ("  GPU      $gpu   VRAM $([int]($vram/1024))GB   드라이버 $driver   CUDA $cudaCap")
    if ($vram -lt 11264) {
        Write-Host "           ※ 12GB 미만이다. 정련은 --L 700~800 으로 낮춰라." -ForegroundColor Yellow
    }
} else {
    Write-Host "  GPU      없음/미검출 — CPU 로도 돌지만 실용 속도가 아니다." -ForegroundColor Yellow
}

# 어떤 CUDA 휠을 받을지. 드라이버가 낮으면 결정을 미룬다.
$wheel = $Cuda
if ($wheel -eq "auto") {
    if (-not $gpu) { $wheel = "cpu" }
    elseif ([version]($driver -replace '^(\d+\.\d+).*', '$1') -ge [version]"528.33") { $wheel = "cu126" }
    else { $wheel = "cpu" }
}
if ($wheel -eq "cpu") {
    # CPU 휠은 CUDA 런타임이 빠져 훨씬 작다. 계획 수치를 바꿔 적는다.
    $PLAN[1] = @{ 항목 = "torch + torchvision (CPU)"; 다운 = 250MB; 디스크 = 900MB }
}
# ★사용자가 -Cuda cpu 를 직접 고른 경우엔 경고할 일이 아니다. auto 가 cpu 로 떨어진 때만.
if ($wheel -eq "cpu" -and $gpu -and $Cuda -eq "auto") {
    Write-Host "           ※ 드라이버 $driver 는 cu126 에 못 미친다(528.33 이상 필요). 드라이버를 올려라." -ForegroundColor Yellow
}

# 파이썬 후보. torch 휠은 cp310~cp315 가 있다. 검증본은 3.12.
$py = $null
if (Have "py") {
    foreach ($v in "3.12", "3.13", "3.11", "3.10", "3.14") {
        $p = (& py "-$v" -c "import sys;print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p) { $py = $p.Trim(); break }
    }
}
if (-not $py -and (Have "python")) {
    $v = (python -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null)
    if ($v -and [version]$v -ge [version]"3.10" -and [version]$v -lt [version]"3.16") {
        $py = (python -c "import sys;print(sys.executable)").Trim()
    }
}
Write-Host ("  python   " + $(if ($py) { "$py" } else { "없음 또는 3.10~3.15 밖" }))

$venvPy = Join-Path $Env "Scripts\python.exe"
$haveVenv = Test-Path $venvPy
Write-Host ("  가상환경 " + $Env + $(if ($haveVenv) { "   (이미 있음)" } else { "   (새로 만듦)" }))

# 이미 갖춰졌는지. 있으면 설치할 게 없다.
$missing = @()
if ($haveVenv) {
    $chk = & $venvPy -c @"
import importlib.util as u
print(','.join(m for m in ('torch','transformers','cv2','PIL','numpy','segmentation_refinement') if u.find_spec(m) is None))
"@ 2>$null
    if ($LASTEXITCODE -eq 0) { $missing = @($chk.Trim() -split ',' | Where-Object { $_ }) }
}

# 모델 캐시
$hfHome = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
$oneformer = Join-Path $hfHome "hub\models--shi-labs--oneformer_ade20k_swin_large"
$cascade = Join-Path $HOME ".segmentation-refinement\model"
$haveOne = Test-Path $oneformer
$haveCas = Test-Path $cascade
Write-Host ("  모델     OneFormer " + $(if ($haveOne) { "있음" } else { "없음" }) +
            "   CascadePSP " + $(if ($haveCas) { "있음" } else { "없음" }))

# ---------------------------------------------------------------- 계획

$todo = @()
if (-not $py) { $todo += $PLAN[0] }
if (-not $haveVenv -or $missing -contains "torch") { $todo += $PLAN[1] }
if (-not $haveVenv -or ($missing | Where-Object { $_ -ne "torch" })) { $todo += $PLAN[2] }
if ($FetchModels -and -not $haveOne) { $todo += $PLAN[3] }
if ($FetchModels -and -not $haveCas) { $todo += $PLAN[4] }

Write-Host ""
if ($todo.Count -eq 0) {
    Write-Host "[설치 계획] 없음 — 필요한 것이 모두 있다." -ForegroundColor Green
    if ($haveVenv) { Write-Host ""; & $venvPy (Join-Path $PSScriptRoot "check_env.py") }
    exit 0
}

# ★Measure-Object -Property 는 해시테이블 키를 못 본다(PS 5.1). 직접 더한다.
$dl = 0; $dk = 0
foreach ($t in $todo) { $dl += $t.다운; $dk += $t.디스크 }
$drive = (Split-Path -Qualifier (Resolve-Path -LiteralPath (Split-Path -Parent $Env) -EA SilentlyContinue).Path)
if (-not $drive) { $drive = (Split-Path -Qualifier $Env) }
$free = try { (Get-PSDrive ($drive -replace ':', '') -EA Stop).Free } catch { -1 }

Write-Host "[설치 계획]"
$todo | ForEach-Object {
    Write-Host ("  {0,-24} 다운 {1,8}   디스크 {2,8}" -f $_.항목, (FmtB $_.다운), (FmtB $_.디스크))
}
Write-Host ("  {0,-24} 다운 {1,8}   디스크 {2,8}" -f "합계", (FmtB $dl), (FmtB $dk))
Write-Host ""
Write-Host "  설치 위치"
Write-Host ("    가상환경   " + $Env)
if ($FetchModels) {
    Write-Host ("    모델       " + (Join-Path $hfHome "hub") + "  ·  " + (Split-Path -Parent $cascade))
}
Write-Host ("    CUDA 휠    " + $wheel)
if ($free -ge 0) {
    Write-Host ("    " + $drive + " 여유 " + (FmtB $free))
    if ($free -lt $dk * 1.3) { Write-Host "    ※ 여유가 빠듯하다." -ForegroundColor Yellow }
}
Write-Host ""

if (-not $Yes) {
    Write-Host "아무것도 바꾸지 않았다. 위 내용에 동의하면 -Yes 를 붙여 다시 실행하라:" -ForegroundColor Cyan
    $cmd = "  .\scripts\bootstrap.ps1 -Yes"
    if ($FetchModels) { $cmd += " -FetchModels" }
    if ($Cuda -ne "auto") { $cmd += " -Cuda $Cuda" }
    Write-Host $cmd -ForegroundColor Cyan
    exit 0
}

# ---------------------------------------------------------------- 실행

if (-not $py) {
    if (-not $winget) { throw "파이썬도 winget 도 없다. python.org 에서 3.12 를 직접 설치하라." }
    Write-Host "`n[1/4] Python 3.12 설치" -ForegroundColor Green
    winget install --id Python.Python.3.12 --scope user --silent `
        --accept-package-agreements --accept-source-agreements
    $py = (& py -3.12 -c "import sys;print(sys.executable)" 2>$null)
    if (-not $py) { throw "설치 후에도 py -3.12 를 못 찾는다. 새 터미널에서 다시 실행하라." }
    $py = $py.Trim()
}

if (-not $haveVenv) {
    Write-Host "`n[2/4] 가상환경 생성  $Env" -ForegroundColor Green
    & $py -m venv $Env
    if ($LASTEXITCODE -ne 0) { throw "venv 생성 실패" }
}
& $venvPy -m pip install --quiet --upgrade pip

Write-Host "`n[3/4] 패키지 설치 ($wheel)  — 수 분 걸린다" -ForegroundColor Green
if ($wheel -eq "cpu") {
    & $venvPy -m pip install torch torchvision
} else {
    & $venvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$wheel"
}
if ($LASTEXITCODE -ne 0) { throw "torch 설치 실패" }
& $venvPy -m pip install transformers opencv-python-headless pillow numpy
if ($LASTEXITCODE -ne 0) { throw "패키지 설치 실패" }
# ★--no-deps 필수. 의존성을 풀면 opencv/torch 를 제 버전으로 갈아엎는다.
#   다만 requests 는 빼면 안 된다 — 모델을 내려받는 데 쓴다. 순수 파이썬이라 안전하다.
#   (실측: --no-deps 만 하면 맨 환경에서 "No module named 'requests'" 로 조용히 죽는다)
& $venvPy -m pip install --no-deps segmentation-refinement
if ($LASTEXITCODE -ne 0) { throw "segmentation-refinement 설치 실패" }
& $venvPy -m pip install requests
if ($LASTEXITCODE -ne 0) { throw "requests 설치 실패" }
& $venvPy -c "import segmentation_refinement" 2>$null
if ($LASTEXITCODE -ne 0) { throw "segmentation-refinement 를 import 할 수 없다 — 정련 단계를 못 쓴다" }

if ($FetchModels) {
    Write-Host "`n[4/4] 모델 내려받기" -ForegroundColor Green
    & $venvPy (Join-Path $PSScriptRoot "check_env.py") --fetch-models
}

Write-Host "`n[검사]" -ForegroundColor Green
& $venvPy (Join-Path $PSScriptRoot "check_env.py")

Write-Host "`n이제 이 인터프리터를 쓰면 된다:" -ForegroundColor Cyan
Write-Host ("  " + $venvPy) -ForegroundColor Cyan
