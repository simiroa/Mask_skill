# -*- coding: utf-8 -*-
"""전수 검증 — 판정 줄로 끝난다. 통과 전에는 확정하지 마라.

검사 항목은 대상 프리셋이 정한다:
  공통   누락 · 읽기실패 · 크기 일치 · 이진성 · 대상 면적 분포 · 잔여 섬/구멍
  zenith 맨윗줄이 대상인가 (ERP 하늘 전용 — 천정은 100% 하늘이어야 한다)
  seam   좌우 끝이 일치하는가 (ERP 전용)

★파이프라인에 단계를 추가할 때마다 앞 단계의 불변식을 다시 검증하라.
  실측에서 타일링의 wrap 불변식(이음새 0.061%)이 정련 단계에서 1.076% 로 깨졌는데,
  검증에 이음새 항목이 없었다면 그대로 넘어갔을 결함이다.

★섬/구멍 판정은 1/2 축소로 한다(문턱 1/4 환산). 전수에 원본 해상도 연결성분을
  돌리면 875장에 1시간이 넘는다 — 순위·유무 판정에는 축소로 충분하다.
★단 축소는 이음새를 과대평가한다(실측 원본 0.037% → 축소 0.162%). 최근접 보간이
  경계 화소를 거칠게 잡기 때문이다. 절대값을 비교할 때는 --full-res 로 재라.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("masks")
    ap.add_argument("--target", default="sky")
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--full-res", action="store_true", help="축소 없이 검사(느리다)")
    ap.add_argument("--island-thr", type=int, default=None,
                    help="비우면 대상 프리셋의 섬 문턱(정책과 검사를 일치시킨다)")
    a = ap.parse_args()

    import cv2, numpy as np
    tgt = T.resolve(a.target)
    checks = tgt["checks"]
    island_thr = a.island_thr if a.island_thr is not None else max(2000, tgt["island_thr"] // 50)
    C.banner("검증", 이미지=a.src, 마스크=a.masks)
    files = C.list_images(a.src, a.suffix)
    bad, area, top, seam, isl, hol, mism = [], [], [], [], 0, 0, 0

    for f in files:
        p = C.mask_path(a.masks, a.src, f, a.suffix)
        if not os.path.exists(p):
            bad.append(("없음", os.path.relpath(f, a.src))); continue
        m = cv2.imread(p, 0)
        if m is None:
            bad.append(("읽기실패", p)); continue
        # ★마스크끼리 비교하면 안 된다. 전부 같은 크기로 틀린 경우가 통과해버린다.
        #   해상도가 섞인 데이터셋은 정상이므로 '원본과 1:1'인지를 파일마다 본다.
        W, H = C.image_size(f)
        if m.shape[:2] != (H, W):
            bad.append((f"크기불일치 마스크{m.shape[1]}x{m.shape[0]}≠원본{W}x{H}",
                        os.path.relpath(f, a.src)))
            mism += 1
            continue
        if not set(np.unique(m).tolist()) <= {0, 255}:
            bad.append(("이진아님", p))
        s = m == 0
        area.append(s.mean())
        if "zenith" in checks:
            top.append((m[0] > 0).mean())
        if "seam" in checks:
            seam.append((m[:, 0] != m[:, -1]).mean())
        h = m if a.full_res else cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2),
                                            interpolation=cv2.INTER_NEAREST)
        thr = island_thr if a.full_res else max(1, island_thr // 4)
        i, o = C.island_stats(h == 0, thr)
        isl += i; hol += o

    n = len(area)
    ok = True
    print(f"[{a.target}] 이미지 {len(files)}장 / 검사통과 마스크 {n}장   문제 {len(bad)}건")
    if bad:
        ok = False
        for b in bad[:6]:
            print("   ", b)
        if len(bad) > 6:
            print(f"    … 외 {len(bad)-6}건")
    if mism:
        print(f"    ※ 원본과 크기가 다른 마스크 {mism}장 — 학습을 조용히 망가뜨린다")
    if not files:
        print("    ※ 이미지가 한 장도 없다 — 경로나 확장자를 확인하라"); ok = False
    if files and n == 0:
        print("    ※ 검사를 통과한 마스크가 없다"); ok = False
    if n:
        ar = np.array(area) * 100
        print(f"    대상 면적  평균 {ar.mean():5.1f}%  범위 {ar.min():.1f}~{ar.max():.1f}%")
        if ar.max() < 1e-6:
            print("    ※ 대상이 한 장도 검출되지 않았다 — 대상·모델·타일 설정을 의심하라"); ok = False
        if ar.std() > 20:
            print("    ※ 면적 편차가 크다 — 실패 프레임이 섞였을 수 있다")
    if top:
        t = np.array(top) * 100
        print(f"    천정(맨윗줄 비대상)  평균 {t.mean():.3f}%  최대 {t.max():.2f}%")
        if t.max() > 5:
            print("    ※ 천정이 비었다 — 화면을 가득 채운 대상을 못 잡는 모델의 전형적 증상"); ok = False
    if seam:
        sm = np.array(seam) * 100
        over = int((sm > 1).sum())
        print(f"    좌우 이음새 불일치  평균 {sm.mean():.3f}%  최대 {sm.max():.2f}%  1% 초과 {over}장")
        if sm.mean() > 0.5:
            print("    ※ 정련 단계가 파노라마를 모른다 — seam.py 를 돌려라"); ok = False
    print(f"    잔여 섬 {isl}개 · 구멍 {hol}개  (문턱 {island_thr:,}px)")
    if tgt["islands"] == "remove" and isl > 0:
        print("    ※ 이 대상은 섬이 없어야 한다 — post.py 의 --speck-island 를 확인하라"); ok = False

    print("\n판정: " + ("통과 — 확정 가능" if ok else "실패 — 확정 보류"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
