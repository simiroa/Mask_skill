# -*- coding: utf-8 -*-
"""후처리 — 과확장 제한 · 티끌 제거 · 마진. GPU 불필요.

세 연산의 순서가 고정이다:
  1) 과확장 제한 — 정련이 성긴 마스크보다 N px 넘게 파고든 곳을 되돌린다
  2) 티끌 제거   — 1)에서 생길 수 있는 조각까지 함께 정리된다
  3) 마진        — 마지막에 얹어야 앞 결과 위에 균일하게 적용된다

★티끌 문턱은 방향별로 다르다(대상 프리셋이 정한다):
  섬(대상 영역 밖에 떠 있는 '비대상' 덩어리 = 마스크상 유지영역 조각) — 하늘에서만 크게 잡는다. 항공영상의 지상
    구조물은 전부 스카이라인으로 이어지므로 하늘 한복판에 떠 있을 것이 원리적으로
    없다(태양·글레어·새뿐). 실측 139프레임에서 태양 제거.
    사람·차·배는 그 자체가 섬이므로 이 정책을 쓰면 대상을 지운다.
  구멍(유지영역 안에 뚫린 '대상' 조각) — 파고라 기둥 사이·크레인 격자로 하늘이
    비치는 경우처럼 실제로 존재하므로 작게 잡는다.

★과확장 제한은 만능이 아니다. 유리 커튼월이 하늘을 반사해 정련이 타워를 얇게
  파먹는 결함은 이 방법으로 못 잡는다 — 쐐기와 정당한 보정이 둘 다 경계에 붙은
  얇은 띠라 구분되지 않는다(실측: 문턱을 낮추면 문제 프레임을 포함해 전부 나빠짐).
  그런 대상은 성긴 마스크를 '넉넉본'으로 따로 내는 편이 낫다.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import targets as T


def limit_growth(m, coarse, max_grow):
    """정련이 대상을 max_grow px 넘게 확장한 부분만 되돌린다. 축소는 건드리지 않는다."""
    import cv2, numpy as np
    if max_grow <= 0:
        return m
    k = np.ones((2 * max_grow + 1, 2 * max_grow + 1), np.uint8)
    return m & cv2.dilate(coarse.astype(np.uint8), k).astype(bool)


def drop_specks(m, thr_hole, thr_island):
    """고립된 작은 성분을 뒤집는다. 가장 큰 성분은 항상 보존.
    m==True 의 작은 성분 = 유지영역에 뚫린 작은 대상 조각(구멍),
    m==False 의 작은 성분 = 대상 영역에 떠 있는 작은 유지 조각(섬, 예: 태양)."""
    import cv2, numpy as np
    out = m.copy()
    for target, thr in ((True, thr_hole), (False, thr_island)):
        if thr <= 0:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats((out == target).astype(np.uint8), 8)
        if n < 2:
            continue
        a = st[1:, cv2.CC_STAT_AREA]
        big = int(np.argmax(a)) + 1
        for i in range(1, n):
            if i != big and a[i - 1] < thr:
                out[lab == i] = not target
    return out


def apply_margin(m, px):
    """양수면 대상이 물러나고(구조물 보존), 음수면 대상이 팽창한다(이동체 여유)."""
    import cv2, numpy as np
    if px == 0:
        return m
    k = np.ones((2 * abs(px) + 1, 2 * abs(px) + 1), np.uint8)
    f = cv2.erode if px > 0 else cv2.dilate
    return f(m.astype(np.uint8), k).astype(bool)


def demo():
    import numpy as np
    z = np.zeros((200, 200), bool)
    base = z.copy(); base[:100] = True

    over = base.copy(); over[100:180, 50:70] = True
    assert not limit_growth(over, base, 30)[170, 60], "30px 넘는 침투가 안 막혔다"
    assert limit_growth(over, base, 30)[110, 60], "허용 범위까지 막아버렸다"
    shrunk = base.copy(); shrunk[80:100] = False
    assert not limit_growth(shrunk, base, 30)[90, 10], "축소 방향을 되돌리면 안 된다"

    t = z.copy(); t[:100] = True
    t[30:70, 30:70] = False        # 대상 영역 속 구멍 1,600px
    t[120:160, 120:160] = True     # 대상 밖의 섬 1,600px
    r = drop_specks(t, thr_hole=1000, thr_island=5000)
    assert r[140, 140] is np.True_ or r[140, 140], "작은 문턱인데 구멍이 지워졌다 — 방향 반대"
    assert r[50, 50], "큰 문턱인데 섬이 안 지워졌다 — 방향 반대"
    keep = drop_specks(t, thr_hole=1000, thr_island=0)
    assert not keep[50, 50], "island_thr=0 인데 섬이 지워졌다"

    m2 = apply_margin(base, 2)
    assert m2[:98].all() and not m2[98:].any(), "양수 마진이 물러나지 않았다"
    m3 = apply_margin(base, -2)
    assert m3[101].all(), "음수 마진이 팽창하지 않았다"
    print("post demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coarse", nargs="?"); ap.add_argument("fine", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--target", default="sky")
    ap.add_argument("--max-grow", type=int, default=5)
    ap.add_argument("--speck", type=int, default=None, help="구멍 문턱. 비우면 프리셋")
    ap.add_argument("--speck-island", type=int, default=None, help="섬 문턱. 비우면 프리셋")
    ap.add_argument("--margin", type=int, default=0, help="양수=대상 후퇴, 음수=대상 팽창")
    ap.add_argument("--suffix", default="_mask")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    demo()
    if not a.coarse:
        return

    import glob, numpy as np
    tgt = T.resolve(a.target)
    hole = tgt["hole_thr"] if a.speck is None else a.speck
    isl = tgt["island_thr"] if a.speck_island is None else a.speck_island
    k, n = map(int, a.shard.split("/"))
    files = sorted(glob.glob(os.path.join(glob.escape(a.fine), "**", "*" + a.suffix + ".png"),
                             recursive=True))[k::n]
    os.makedirs(a.dst, exist_ok=True)
    print(f"{len(files):,}장  대상 {a.target}  과확장 {a.max_grow}px  "
          f"구멍 {hole:,}px  섬 {isl:,}px  마진 {a.margin:+d}px", flush=True)
    t0, done, delta, broken, no_coarse = time.time(), 0, [], 0, 0
    for f in files:
        rel = os.path.relpath(f, a.fine)
        out = os.path.join(a.dst, rel)
        if os.path.exists(out):
            continue
        cm = os.path.join(a.coarse, rel)
        fine = C.read_mask(f)
        if fine is None:
            print("손상된 마스크, 건너뜀:", f, flush=True); broken += 1; continue
        crs = C.read_mask(cm) if os.path.exists(cm) else None
        if crs is None:
            crs = fine; no_coarse += 1
        before = fine.mean()
        m = limit_growth(fine, crs, a.max_grow)
        m = drop_specks(m, hole, isl)
        m = apply_margin(m, a.margin)
        C.write_mask(out, m)
        delta.append(m.mean() - before)
        done += 1
        if done % 100 == 0:
            e = time.time() - t0
            print(f"  {done}/{len(files)}  {e/done:.2f}s/장", flush=True)
    d = np.array(delta) * 100 if delta else np.zeros(1)
    print(f"완료 {done}장  {(time.time()-t0)/60:.1f}분   "
          f"대상 면적 변화 평균 {d.mean():+.3f}%p  최대감소 {d.min():+.3f}%p", flush=True)
    if done == 0:
        print("  ※ 0장 처리됐다 — 출력 폴더에 이미 파일이 있거나(재개) 입력이 비었다.", flush=True)
    if no_coarse:
        print(f"  ※ 성긴 마스크가 없어 과확장 제한이 무효였던 프레임 {no_coarse}장", flush=True)
    if broken:
        print(f"  ※ 손상되어 건너뛴 마스크 {broken}장 — 지우고 다시 돌려라", flush=True)


if __name__ == "__main__":
    main()
