# -*- coding: utf-8 -*-
"""환경 검사 — 무엇이 있고 무엇이 없는지, 없으면 어떤 명령으로 넣는지.

★기본은 제안만 한다. 사용자 환경을 임의로 바꾸지 않는다.
  --install 을 명시해야 실제로 설치한다. 그때도 용량·경로를 먼저 보이고 --yes 를 요구한다.

파이썬이 아예 없는 맨몸 윈도우는 이 스크립트가 돌지 않는다 — scripts/bootstrap.ps1 을 써라.

  check_env.py                     # 검사만
  check_env.py --install           # 설치 계획(용량·경로) 출력, 바꾸지 않음
  check_env.py --install --yes     # 실제 설치
  check_env.py --fetch-models      # 모델을 미리 받는다(첫 실행 대기 없음)
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

NEED = [
    ("torch",                    "torch",                    "필수", None),
    ("transformers",             "transformers",             "필수", "pip install transformers"),
    ("cv2",                      "opencv-python-headless",   "필수", "pip install opencv-python-headless"),
    ("PIL",                      "pillow",                   "필수", "pip install pillow"),
    ("numpy",                    "numpy",                    "필수", "pip install numpy"),
    ("segmentation_refinement",  "segmentation-refinement",  "정련", "pip install --no-deps segmentation-refinement"),
    # ★--no-deps 로 넣으므로 requests 가 안 따라온다. 모델 다운로드에 필요하다.
    ("requests",                 "requests",                 "정련", "pip install requests"),
]

# 설치 용량 — 다운로드는 압축 기준, 디스크는 풀린 기준. 실측(py3.12 / cu126).
SIZE = {
    "torch":                   (2.5 * 2**30, 3.9 * 2**30),
    "transformers":            (120 * 2**20, 900 * 2**20),
    "opencv-python-headless":  (40 * 2**20, 120 * 2**20),
    "pillow":                  (3 * 2**20, 10 * 2**20),
    "numpy":                   (8 * 2**20, 30 * 2**20),
    "segmentation-refinement": (1 * 2**20, 2 * 2**20),
    "requests":                (1 * 2**20, 3 * 2**20),
}
MODELS = [
    ("OneFormer Swin-L", 1.64 * 2**30, "~/.cache/huggingface/hub/"),
    ("CascadePSP", 271 * 2**20, "~/.segmentation-refinement/model"),
]


def fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def do_install(pkgs, yes):
    """★설치는 되돌리기 번거롭다. 용량·경로를 먼저 보이고 --yes 를 받아야 실행한다."""
    dl = sum(SIZE.get(p, (0, 0))[0] for p, _ in pkgs)
    dk = sum(SIZE.get(p, (0, 0))[1] for p, _ in pkgs)
    free = shutil.disk_usage(os.path.dirname(sys.executable)).free
    print()
    print("[설치 계획]")
    for pkg, _ in pkgs:
        d, k = SIZE.get(pkg, (0, 0))
        print(f"  {pkg:26s} 다운 {fmt(d):>8s}   디스크 {fmt(k):>8s}")
    print(f"  {'합계':26s} 다운 {fmt(dl):>8s}   디스크 {fmt(dk):>8s}")
    print(f"  설치 위치      {os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))}")
    print(f"  드라이브 여유  {fmt(free)}")
    if free < dk * 1.3:
        print("  ※ 여유가 빠듯하다.")
    if not yes:
        print()
        print("아무것도 바꾸지 않았다. 동의하면 --yes 를 붙여 다시 실행하라:")
        print(f"  {sys.executable} {os.path.abspath(__file__)} --install --yes")
        return 1
    for pkg, cmd in pkgs:
        if not cmd:
            print(f"  건너뜀 {pkg} — GPU 에 맞는 빌드를 직접 골라야 한다."
                  " bootstrap.ps1 이 드라이버를 보고 정해준다.")
            continue
        args = cmd.split()[2:]                      # "pip install …" 뒤만 넘긴다
        print()
        print(f"  설치 {pkg}", flush=True)
        if subprocess.run([sys.executable, "-m", "pip"] + args).returncode:
            print(f"  실패: {pkg}")
            return 1
    print()
    print("설치 완료. 인자 없이 다시 실행해 검사하라.")
    return 0


def fetch_models():
    """첫 실행 때 자동으로 받지만, 미리 받아두면 처리 도중 대기가 없다."""
    print()
    print(f"[모델 내려받기]  총 {fmt(sum(s for _, s, _ in MODELS))}")
    for name, size, where in MODELS:
        print(f"  {name:20s} {fmt(size):>8s}  {where}")
    try:
        from transformers import AutoProcessor, AutoModelForUniversalSegmentation
        mid = "shi-labs/oneformer_ade20k_swin_large"
        AutoProcessor.from_pretrained(mid)
        AutoModelForUniversalSegmentation.from_pretrained(mid)
        print("  OneFormer OK")
    except Exception as e:
        print(f"  OneFormer 실패: {str(e)[:120]}")
        return 1
    try:
        import segmentation_refinement as sr
        sr.Refiner(device="cpu")                    # 없으면 여기서 받는다
        print("  CascadePSP OK")
    except Exception as e:
        print(f"  CascadePSP 실패: {str(e)[:120]}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="빠진 패키지 설치(먼저 계획만 출력)")
    ap.add_argument("--yes", action="store_true", help="--install 과 함께 주면 실제로 설치한다")
    ap.add_argument("--fetch-models", action="store_true", help="모델을 미리 받는다")
    a = ap.parse_args()

    print(f"python  {sys.version.split()[0]}   {sys.executable}")
    missing, optional = [], []
    for mod, pkg, kind, cmd in NEED:
        ok = have(mod)
        ver = ""
        if ok:
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "") or ""
            except Exception as e:
                ok, ver = False, f"import 실패: {str(e)[:60]}"
        print(f"  {'OK ' if ok else '없음'}  {mod:24s} {ver:12s} [{kind}]")
        if not ok:
            (missing if kind == "필수" else optional).append((pkg, cmd))

    if have("torch"):
        import torch
        cu = torch.cuda.is_available()
        print()
        print(f"torch {torch.__version__}   CUDA {'사용가능' if cu else '없음'}")
        if cu:
            free, total = torch.cuda.mem_get_info()
            print(f"  {torch.cuda.get_device_name(0)}   VRAM {total/2**30:.1f}GB "
                  f"(현재 여유 {free/2**30:.1f}GB)")
            if total / 2**30 < 11:
                print("  주의: 12GB 미만이면 CascadePSP L=900 이 빠듯하다. L 을 700~800 으로 낮춰라.")
        else:
            print("  CPU 로도 돌지만 실용 속도가 아니다.")

    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        print(f"  nvidia-smi 실사용: {r.stdout.strip()}")
        print("  ★VRAM 은 nvidia-smi 로 재라. torch.cuda.max_memory_allocated() 는"
              " 예약블록·컨텍스트를 빼먹어 3GB 가까이 과소보고한다.")

    if missing or optional:
        print()
        print("설치 제안 (실행 여부는 사용자가 결정):")
        for pkg, cmd in missing + optional:
            print(f"  {cmd or pkg + ' 는 GPU 에 맞는 빌드를 직접 설치해야 한다'}")
        if any(p == "segmentation-refinement" for p, _ in optional):
            print("  ※ segmentation-refinement 는 --no-deps 로 넣어라."
                  " 의존성을 풀면 opencv/torch 를 갈아엎는다.")
        if a.install:
            return do_install(missing + optional, a.yes)
        print("  ※ 여기서 설치하려면 --install (계획만) → --install --yes (실행).")
    else:
        print()
        print("필요한 것이 모두 있다.")

    if a.fetch_models:
        return fetch_models()
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
