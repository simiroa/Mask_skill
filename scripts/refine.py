# -*- coding: utf-8 -*-
"""패스2 — CascadePSP 로 경계만 정련. 의미 판단은 건드리지 않는다.

실측(7776x3888, 12GB 카드): 3px 이내 67.9% -> 97.7%, 대상 면적은 +0.3%p 로 사실상 불변.

L 은 국소 정련 창 크기. 손잡이가 아니라 절벽이다:
    L= 900  12.5s/장  카드 ~7,000MiB   3px내 97.7%   ← 유일한 실용값
    L=1200  13.7s/장       11,580MiB        97.4%
    L=1400  94.6s/장       11,968MiB   ← 카드가 차는 순간 7배
    L=1800 290.2s/장       13,000MiB
품질은 900~1200 구간에서 평평하다. 올려서 얻을 게 없다.
VRAM 이 12GB 미만이면 L 을 700~800 으로 낮춰라.

대상이 없는 프레임(마스크가 전부 255)은 건너뛴다 — 다듬을 경계가 없고,
없는 경계를 만들어낼 수 있다.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("coarse"); ap.add_argument("dst")
    ap.add_argument("--L", type=int, default=900)
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--min-area", type=float, default=1e-5,
                    help="대상 면적이 이 비율 미만이면 정련을 건너뛰고 그대로 복사")
    a = ap.parse_args()

    import cv2, numpy as np, shutil
    import segmentation_refinement as refine

    k, n = map(int, a.shard.split("/"))
    files = C.list_images(a.src, a.suffix)[k::n]
    os.makedirs(a.dst, exist_ok=True)
    C.banner("패스2 경계정련", 입력=a.src, 성긴=a.coarse, 출력=a.dst)
    C.plan_report(files, a.dst, "정련")
    ref = refine.Refiner(device="cuda")
    pr = C.Progress(len(files), every=10)
    t0, done, skipped, broken = time.time(), 0, 0, 0
    for f in files:
        out = C.mask_path(a.dst, a.src, f, a.suffix)
        if os.path.exists(out):
            continue
        cm = C.mask_path(a.coarse, a.src, f, a.suffix)
        if not os.path.exists(cm):
            print("성긴 마스크 없음:", os.path.relpath(f, a.src), flush=True); continue
        tgt = C.read_mask(cm)
        if tgt is None:
            print("손상된 성긴 마스크:", cm, flush=True); broken += 1; continue
        if tgt.mean() < a.min_area or (1 - tgt.mean()) < a.min_area:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            shutil.copyfile(cm, out); skipped += 1; continue
        img = C.load_bgr(f)
        soft = ref.refine(img, (tgt * 255).astype(np.uint8), fast=False, L=a.L)
        C.write_mask(out, soft > 127)
        done += 1
        pr.tick()
    pr.n = done
    pr.finish(a.dst, f"· 건너뜀 {skipped}장 · 손상 {broken}장")


if __name__ == "__main__":
    main()
