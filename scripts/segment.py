# -*- coding: utf-8 -*-
"""패스1 — 시맨틱 분할로 성긴 마스크. 0 = 대상(제외), 255 = 유지.

--only <목록파일> 로 대상이 있는 프레임만 돌릴 수 있다(상대경로 한 줄에 하나).
대상이 없는 프레임은 마스크가 전부 255 라 모델을 돌리는 게 낭비일 뿐 아니라,
없는 대상을 만들어낼 위험도 있다.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--target", default="sky")
    ap.add_argument("--model", default="oneformer", choices=list(C.MODELS))
    ap.add_argument("--tile", default="", help="COLSxROWS. 비우면 대상 프리셋 값")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--wrap", action="store_true", default=None,
                    help="가로를 360°로 감는다(ERP 전용). 기본은 이미지 종횡비로 자동 판정")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.add_argument("--infer-width", type=int, default=1024,
                    help="segformer 만 유효. oneformer/mask2former 는 프로세서가 고정한다")
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fill-empty", action="store_true",
                    help="대상이 없다고 알려진 프레임에 전부 255 마스크를 직접 쓴다")
    ap.add_argument("--only", default="", help="처리할 상대경로 목록 파일")
    a = ap.parse_args()

    import numpy as np

    tgt = T.resolve(a.target)
    cols, rows = ([int(x) for x in a.tile.lower().split("x")] if a.tile else tgt["tile"])
    k, n = map(int, a.shard.split("/"))

    files = C.list_images(a.src, a.suffix)
    if a.only:
        want = {l.strip().replace("/", os.sep) for l in open(a.only, encoding="utf-8") if l.strip()}
        files = [f for f in files if os.path.relpath(f, a.src) in want]
    if a.limit:
        files = files[:a.limit]          # ★샤딩보다 먼저 — 뒤에 두면 전체가 limit*N 장이 된다
    files = files[k::n]
    C.banner("패스1 분할", 입력=a.src, 출력=a.dst)
    print(f"  {len(files):,}장 (샤드 {k}/{n})  대상 {a.target}{tgt['classes']}  "
          f"타일 {cols}x{rows}  wrap={'auto' if a.wrap is None else a.wrap}  "
          f"모델 {a.model}", flush=True)
    C.plan_report(files, a.dst, "패스1")

    if a.fill_empty:
        done = 0
        for f in files:
            out = C.mask_path(a.dst, a.src, f, a.suffix)
            if os.path.exists(out):
                continue
            W, H = C.image_size(f)             # 화소를 디코드하지 않는다(EXIF 적용)
            C.write_mask(out, np.zeros((H, W), bool))
            done += 1
        n_, b_ = C.dir_stats(a.dst)
        print(f"빈 마스크 {done}장 생성 -> {os.path.abspath(a.dst)}  "
              f"{n_:,}개 {C.fmt_bytes(b_)}", flush=True)
        return

    os.makedirs(a.dst, exist_ok=True)
    model = C.SkyModel(a.model)
    stats = open(os.path.join(a.dst, f"stats.{k}.jsonl"), "a", encoding="utf-8")
    pr = C.Progress(len(files), every=25)
    done = 0
    for f in files:
        out = C.mask_path(a.dst, a.src, f, a.suffix)
        if os.path.exists(out):
            continue
        arr = C.load_rgb(f)
        H_, W_ = arr.shape[:2]
        # ★wrap 기본값은 이미지마다 종횡비로 정한다. 일반 프레임에 감기를 켜면
        #   화면 왼쪽 끝을 오른쪽 끝과 같은 장면으로 착각한다.
        wrap = (abs(W_ / H_ - 2.0) < C.ERP_RATIO_TOL) if a.wrap is None else a.wrap
        m = model.target(arr, tgt["classes"], cols, rows, a.overlap, wrap, a.infer_width)
        C.write_mask(out, m)
        stats.write(json.dumps({"f": os.path.relpath(f, a.src).replace(os.sep, "/"),
                                "area": round(float(m.mean()), 6),
                                "size": list(arr.shape[:2][::-1])}) + "\n")
        stats.flush()
        done += 1
        pr.tick()
    stats.close()
    pr.finish(a.dst)


if __name__ == "__main__":
    main()
