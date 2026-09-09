# -*- coding: utf-8 -*-
"""ERP 전용 — 좌우 이음새 복구. 파노라마가 아니면 실행하지 마라.

왜 필요한가: 정련기(CascadePSP)는 파노라마인 걸 모른다. 왼쪽 끝과 오른쪽 끝을
각각 따로 다듬어 서로 어긋난다. 실측 875장에서 평균 0.061% → 1.076%, 최대 13.04%,
1% 초과가 14장 → 236장으로 늘었다. 타일링에는 wrap 을 넣었는데 정련 단계에
같은 고려를 안 한 결과다 — 앞 단계의 전제는 뒤 단계로 자동으로 따라가지 않는다.

방법: 이음새가 중앙에 오도록 감은 뒤 그 둘레 띠만 다시 정련해 중앙부만 붙인다.
띠 가장자리에서 500px 안쪽에 붙이므로 두 정련본이 같은 국소 맥락을 본다 —
실측상 붙인 경계의 불연속이 임의의 열과 같은 수준(0.00~0.18% vs 0.03~0.08%)이라
새 이음새가 생기지 않는다. 전량 재정련의 1/3 비용.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STRIP, PASTE = 2400, 1400


def paste_window(W, strip=STRIP, paste=PASTE):
    """감은 좌표계에서 (정련 띠, 붙일 범위). 붙일 범위는 띠 안에 충분히 들어가야 한다."""
    sh = W // 2
    x0, x1 = sh - strip // 2, sh + strip // 2
    p0, p1 = sh - paste // 2, sh + paste // 2
    assert x0 < p0 < p1 < x1, "붙일 범위가 띠 가장자리에 너무 가깝다"
    assert 0 <= x0 and x1 <= W, "띠가 이미지를 벗어난다 — 폭이 좁으면 STRIP 을 줄여라"
    assert p0 - x0 >= 400 and x1 - p1 >= 400, "여유가 400px 미만이면 맥락이 부족하다"
    return sh, (x0, x1), (p0, p1)


def demo():
    sh, (x0, x1), (p0, p1) = paste_window(7776)
    assert (sh, x0, x1, p0, p1) == (3888, 2688, 5088, 3188, 4588)
    print("seam demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("coarse", nargs="?")
    ap.add_argument("fine", nargs="?"); ap.add_argument("dst", nargs="?")
    ap.add_argument("--L", type=int, default=900)
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    demo()
    if not a.src:
        return

    import cv2, numpy as np
    import segmentation_refinement as refine

    k, n = map(int, a.shard.split("/"))
    files = C.list_images(a.src, a.suffix)[k::n]
    os.makedirs(a.dst, exist_ok=True)
    C.banner("이음새 복구", 입력=a.src, 성긴=a.coarse, 정련=a.fine, 출력=a.dst)
    C.plan_report(files, a.dst, "이음새")
    ref = refine.Refiner(device="cuda")
    pr = C.Progress(len(files), every=25)
    done = 0
    for f in files:
        out = C.mask_path(a.dst, a.src, f, a.suffix)
        if os.path.exists(out):
            continue
        cm = C.mask_path(a.coarse, a.src, f, a.suffix)
        fm = C.mask_path(a.fine, a.src, f, a.suffix)
        if not (os.path.exists(cm) and os.path.exists(fm)):
            print("입력 없음:", os.path.relpath(f, a.src), flush=True); continue
        img = C.load_bgr(f)
        H, W = img.shape[:2]
        if abs(W / H - 2.0) > C.ERP_RATIO_TOL:
            raise SystemExit(f"ERP 가 아니다(종횡비 {W/H:.2f}). 이 단계는 파노라마 전용이다.")
        sh, (x0, x1), (p0, p1) = paste_window(W)
        c0, f0 = C.read_mask(cm), C.read_mask(fm)
        if c0 is None or f0 is None:
            print("손상된 마스크:", os.path.relpath(f, a.src), flush=True); continue
        crs = np.roll(c0, sh, axis=1)
        fin = np.roll(f0, sh, axis=1)
        soft = ref.refine(np.roll(img, sh, axis=1)[:, x0:x1],
                          (crs[:, x0:x1] * 255).astype(np.uint8), fast=False, L=a.L)
        fin[:, p0:p1] = (soft > 127)[:, p0 - x0:p1 - x0]
        C.write_mask(out, np.roll(fin, -sh, axis=1))
        done += 1
        pr.tick()
    pr.finish(a.dst)


if __name__ == "__main__":
    main()
