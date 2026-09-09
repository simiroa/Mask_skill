# -*- coding: utf-8 -*-
"""환경 검사 — 무엇이 있고 무엇이 없는지, 없으면 어떤 명령으로 넣는지.

설치를 직접 하지 않는다. 사용자 환경을 임의로 바꾸지 않고 제안만 한다.
"""
import importlib.util
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
]


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def main():
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
        print(f"\ntorch {torch.__version__}   CUDA {'사용가능' if cu else '없음'}")
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
        print("\n설치 제안 (실행 여부는 사용자가 결정):")
        for pkg, cmd in missing + optional:
            print(f"  {cmd or pkg + ' 는 환경에 맞는 빌드를 직접 설치해야 한다'}")
        if any(p == "segmentation-refinement" for p, _ in optional):
            print("  ※ segmentation-refinement 는 --no-deps 로 넣어라."
                  " 의존성을 풀면 opencv/torch 를 갈아엎는다.")
    else:
        print("\n필요한 것이 모두 있다.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
