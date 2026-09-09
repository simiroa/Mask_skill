# -*- coding: utf-8 -*-
"""오버레이 비교 렌더 — 모든 비교 그림이 같은 규약을 쓰도록 강제한다.

★설정 비교는 '두 결과가 가장 많이 갈리는 구간'을 봐야 한다.
  무작위 위치나 미리 정한 좌표로는 0.1% 차이를 놓친다 — 실측에서 두 설정의
  차이 화소가 0.116% 뿐이었는데 그 차이가 건물 하나에 통째로 몰려 있었고,
  주 지표(3px 이내)는 95.6% = 95.6% 로 같다고 말하고 있었다.

★수치만으로 판정하지 마라. 주 지표는 '어떤' 에지인지 모른다. Depth 계열이
  3px 이내 75% 를 찍고도 실제로는 엉뚱한 질감선을 따라다닌 전례가 있다.

사용:
  compare.py <이미지> --masks A=dirA B=dirB --pick 3        # 설정 비교
  compare.py <이미지> --masks M=dir --pick 6 --mode detail  # 단일 검토
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--masks", nargs="+", required=True, help="라벨=폴더 …")
    ap.add_argument("--out", default="compare")
    ap.add_argument("--pick", type=int, default=3, help="렌더할 프레임 수")
    ap.add_argument("--frames", default="", help="특정 상대경로들(콤마). 비우면 자동 선택")
    ap.add_argument("--mode", default="diff", choices=["diff", "detail"],
                    help="diff=가장 많이 갈리는 구간, detail=경계가 가장 복잡한 구간")
    ap.add_argument("--box", default="1500x400", help="크롭 WxH")
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image

    sets = []
    for spec in a.masks:
        label, _, path = spec.partition("=")
        assert path, f"라벨=폴더 형식이어야 한다: {spec}"
        sets.append((label, path))
    cw, ch = (int(x) for x in a.box.lower().split("x"))
    os.makedirs(a.out, exist_ok=True)

    files = C.list_images(a.src, a.suffix)
    if a.frames:
        want = {f.strip().replace("/", os.sep) for f in a.frames.split(",")}
        files = [f for f in files if os.path.relpath(f, a.src) in want]
    else:
        # 자동 선택: 마스크가 모두 존재하고 대상이 실제로 있는 프레임 중에서
        ok = []
        for f in files:
            ps = [C.mask_path(p, a.src, f, a.suffix) for _, p in sets]
            if all(os.path.exists(p) for p in ps):
                ok.append(f)
        random.seed(a.seed)
        files = random.sample(ok, min(a.pick * 4, len(ok))) if ok else []

    made = 0
    for f in files:
        if made >= a.pick:
            break
        masks = []
        for label, p in sets:
            m = C.read_mask(C.mask_path(p, a.src, f, a.suffix))
            if m is None:
                break
            masks.append((label, m))
        if len(masks) != len(sets):
            continue
        if not a.frames and masks[0][1].mean() < 0.005:
            continue                       # 대상이 거의 없는 프레임은 볼 게 없다
        img = C.load_rgb(f)
        H, W = img.shape[:2]
        arrs = [m for _, m in masks]
        x0, y0 = (C.crop_most_different(arrs, W, H, cw, ch) if a.mode == "diff"
                  else C.crop_most_detailed(arrs[0], W, H, cw, ch))
        entries = [("원본", None)] + [(lb, m) for lb, m in masks]
        sheet = C.contact_sheet(img, entries, (x0, y0, min(cw, W), min(ch, H)),
                                zoom=a.zoom, vertical=(cw > 2 * ch))
        stem = os.path.splitext(os.path.relpath(f, a.src))[0].replace(os.sep, "_")
        out = os.path.join(a.out, f"{stem}.jpg")
        Image.fromarray(sheet).save(out, quality=94)
        d = ""
        if len(arrs) > 1:
            diff = np.zeros_like(arrs[0])
            for m in arrs[1:]:
                diff |= (arrs[0] != m)
            d = f"  차이 {diff.mean()*100:.3f}%"
            for lb, m in masks:
                d += f"  {lb} {m.mean()*100:.2f}%"
        print(f"{stem}  크롭({x0},{y0}){d}", flush=True)
        made += 1
    print(f"\n{made}장 저장 -> {a.out}")
    if made:
        print("판정은 수치가 아니라 이 그림으로 하라. 수치는 보조다.")


if __name__ == "__main__":
    main()
