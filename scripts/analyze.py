# -*- coding: utf-8 -*-
"""데이터셋 분석 → 설정 추천. 모델을 돌리지 않는다(메타데이터만).

무엇을 판별하나:
  - ERP(등장방형)인가 일반 프레임인가  → wrap / 이음새 단계 필요 여부
  - 해상도 분포가 섞여 있는가          → 배치 시간 편차 경고
  - 화소 수                            → 정련 비용과 VRAM 추정
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import targets as T


def classify(sizes):
    """(모드, 근거). ERP 판정은 종횡비 2:1 이 지배적인지로 한다."""
    ratios = [w / h for w, h in sizes]
    erp = sum(1 for r in ratios if abs(r - 2.0) < C.ERP_RATIO_TOL)
    if erp >= 0.9 * len(ratios):
        return "erp", f"종횡비 2.00 이 {erp}/{len(ratios)}"
    return "flat", f"종횡비 {min(ratios):.2f}~{max(ratios):.2f} (2:1 아님)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--target", default="sky")
    ap.add_argument("--sample", type=int, default=40)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    files = C.list_images(a.src)
    if not files:
        raise SystemExit(f"이미지가 없다: {a.src}")
    sizes, sub = [], collections.Counter()
    idx = np.linspace(0, len(files) - 1, min(a.sample, len(files))).astype(int)
    for f in [files[i] for i in sorted(set(idx.tolist()))]:
        sizes.append(C.image_size(f))
        sub[os.path.relpath(os.path.dirname(f), a.src)] += 1
    cnt = collections.Counter(sizes)

    tgt = T.resolve(a.target)
    mode, why = classify(sizes)
    mp = sum(w * h for w, h in sizes) / len(sizes) / 1e6

    C.banner("소스 분석", 이미지=a.src)
    print(f"이미지 {len(files):,}장   표본 {len(sizes)}장")
    print(f"해상도: " + ", ".join(f"{w}x{h} x{n}" for (w, h), n in cnt.most_common(4)))
    if len(cnt) > 1:
        print("  ※ 해상도가 섞여 있다 — 프레임당 시간 편차가 크다. 진행률 추정을 믿지 마라.")
    if len(sub) > 1:
        print(f"하위 폴더: {dict(list(sub.items())[:6])}")
    print(f"판정: {mode.upper()}  ({why})")
    print(f"대상: {a.target} — {tgt['label']}, ADE20K 클래스 {tgt['classes']}")
    print(f"  {tgt['note']}")

    wrap = (mode == "erp")
    cols, rows = tgt["tile"]
    print("\n추천 설정")
    print(f"  --target {a.target}  --tile {cols}x{rows}  {'--wrap' if wrap else '--no-wrap'}")
    if not wrap:
        print("     일반 프레임이다. 가로를 감으면 화면 좌우 끝을 같은 장면으로 착각한다.")
    if "seam" in tgt["checks"] and wrap:
        print("  이음새 복구 단계 필요 (seam.py) — 정련이 파노라마인 걸 모른다")
    elif not wrap:
        print("  이음새 복구 불필요 (파노라마 아님)")
    if tgt["islands"] == "remove":
        print(f"  섬 제거 켬 (--speck-island {tgt['island_thr']}) — 떠 있는 조각은 오검출로 본다")
    else:
        print("  섬 제거 끔 — 이 대상은 그 자체가 고립된 섬일 수 있다")

    est = mp / 30.2                      # 7776x3888 = 30.2MP 를 1.0 으로 본 상대비용
    print(f"\n비용 추정 (7776x3888 실측 기준 상대)")
    print(f"  평균 {mp:.1f}MP  ->  패스1 약 {3.5*est:.1f}s/장, 정련 약 {12.5*est:.1f}s/장")
    print(f"  {len(files):,}장 전량: 패스1 {C.fmt_dur(3.5*est*len(files))}, "
          f"정련 {C.fmt_dur(12.5*est*len(files))}, "
          f"이음새 {C.fmt_dur(4.4*est*len(files))}(ERP만), 후처리 {C.fmt_dur(1.0*est*len(files))}")
    per = C.mask_bytes_per_frame(sizes)
    stage = per * len(files)
    n_stage = 4 if wrap else 3
    print("")
    print(f"용량 추정   마스크 1장 약 {C.fmt_bytes(per)}  ->  단계당 {C.fmt_bytes(stage)}")
    print(f"  중간 산출물 {n_stage}단계 유지 시 총 {C.fmt_bytes(stage * n_stage)}  "
          f"(출력 예정 드라이브 여유 {C.fmt_bytes(C.free_space(os.path.abspath(a.src)))})")
    print("  ※ 중간 산출물을 지우지 마라 — 설정을 바꿀 때 모델 재실행 없이 다시 만든다.")
    print("  ※ 정련은 경계가 긴 프레임에서 5배까지 튄다(실측 12s~64s). 평균만 믿지 마라.")
    print("  ※ 대상이 없는 프레임은 정련을 건너뛰어라 — 다듬을 경계가 없다.")


if __name__ == "__main__":
    main()
