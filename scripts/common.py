# -*- coding: utf-8 -*-
"""semantic-mask 공용 모듈 — 타일링·추론·마스크 입출력·오버레이·지표.

마스크 규약(전 스크립트 공통): 0 = 대상(손실에서 제외), 255 = 유지.
"""
import glob
import os

import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SKY_CLASS = 2                    # ADE20K 의 sky
ERP_RATIO_TOL = 0.05             # 종횡비가 2:1 에서 이만큼 안이면 ERP 로 본다(전 스크립트 공유)
MODELS = {
    "oneformer":   "shi-labs/oneformer_ade20k_swin_large",
    "mask2former": "facebook/mask2former-swin-large-ade-semantic",
    "segformer":   "nvidia/segformer-b4-finetuned-ade-512-512",
}
# 오버레이 렌더 규약 — 모든 비교 이미지가 같아 보이도록 여기서만 정한다
OV_ALPHA = 0.42                  # 원본을 남기는 비율
OV_COLOR = (255, 45, 45)         # 하늘(제외) 영역 표시색 RGB
OV_EDGE = (255, 255, 0)          # 경계선 색


# ---------------------------------------------------------------- 파일

def list_images(root, suffix="_mask"):
    """마스크 출력물을 입력으로 다시 먹지 않도록 걸러서 나열한다."""
    out = []
    # ★glob.escape: 폴더명의 [ ] * ? 가 패턴으로 먹혀 0장이 나오는 사고를 막는다
    for p in sorted(glob.glob(os.path.join(glob.escape(root), "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in EXTS and not (suffix and stem.endswith(suffix)):
            out.append(p)
    # 확장자만 다른 동명 파일은 마스크 경로가 겹친다 — 조용히 덮이므로 알린다
    seen = {}
    for p in out:
        k = os.path.splitext(p)[0]
        seen.setdefault(k, []).append(p)
    dup = [v for v in seen.values() if len(v) > 1]
    if dup:
        print(f"경고: 확장자만 다른 동명 파일 {len(dup)}쌍 — 마스크가 서로 덮인다. "
              f"예: {[os.path.basename(x) for x in dup[0]]}")
    return out


def mask_path(dst, src_root, img_path, suffix="_mask"):
    """<원본이름><suffix>.png. 널리 쓰이는 규약이라 바꾸지 않는다.
    ※ 같은 폴더에 a.jpg 와 a.png 가 함께 있으면 마스크가 충돌한다 —
      list_images 가 시작할 때 경고한다."""
    rel = os.path.relpath(img_path, src_root)
    return os.path.join(dst, os.path.splitext(rel)[0] + suffix + ".png")


def load_rgb(path):
    """EXIF Orientation 을 적용해 RGB 배열로 읽는다.

    ★PIL 은 EXIF 회전을 무시하고 cv2.imread 는 적용한다. 둘을 섞어 쓰면
      같은 파일에서 가로세로가 뒤바뀐다(드론·폰 JPEG 에 흔하다). 여기로 통일한다.
    """
    import numpy as np
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def load_bgr(path):
    """cv2 계열(정련기)에 넘길 BGR 배열. load_rgb 와 같은 방향을 보장한다."""
    import cv2
    return cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2BGR)


def image_size(path):
    """EXIF 적용 후 (W, H). 화소를 디코드하지 않는다."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    return im.size


def read_mask(path):
    """반환: 하늘 불리언 (True = 하늘)."""
    import cv2
    m = cv2.imread(path, 0)
    return None if m is None else (m == 0)


def write_mask(path, sky):
    """★임시 파일에 쓰고 교체한다. cv2.imwrite 는 원자적이지 않아,
    쓰기 도중 중단되면 반쪽 PNG 가 남고 재개 로직이 그걸 '완료'로 보고 영원히 건너뛴다."""
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.png"
    ok = cv2.imwrite(tmp, np.where(sky, 0, 255).astype(np.uint8))
    if not ok:
        raise IOError(f"마스크 쓰기 실패: {path}")
    os.replace(tmp, path)


# ---------------------------------------------------------------- 보고 형식
# 모든 스크립트가 같은 형식으로 절대경로·용량·시간을 낸다.

def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def fmt_dur(sec):
    if sec < 60:
        return f"{sec:.0f}초"
    if sec < 3600:
        return f"{sec/60:.1f}분"
    return f"{sec/3600:.1f}시간"


def dir_stats(path, pattern="**/*"):
    """(파일 수, 총 바이트). 없으면 (0, 0)."""
    if not path or not os.path.isdir(path):
        return 0, 0
    n = t = 0
    for f in glob.glob(os.path.join(glob.escape(path), pattern), recursive=True):
        if os.path.isfile(f):
            n += 1
            t += os.path.getsize(f)
    return n, t


def free_space(path):
    import shutil
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or ".").free
    except OSError:
        return -1


def banner(title, **paths):
    """절대경로를 항상 보여준다 — 어디에 쓰는지 헷갈리는 게 가장 흔한 사고다."""
    print(f"[{title}]")
    for k, v in paths.items():
        if not v:
            continue
        ap = os.path.abspath(v)
        n, b = dir_stats(ap)
        extra = f"   ({n:,}개, {fmt_bytes(b)})" if n else ("   (없음/빈 폴더)" if os.path.isdir(ap) else "   (새로 만듦)")
        pad = " " * max(1, 9 - sum(2 if ord(c) > 0x2000 else 1 for c in k))
        print(f"  {k}{pad}{ap}{extra}")


# 단계별 프레임당 비용 — 7776x3888(30.2MP) 12GB 카드 실측 기준. 화소 수에 비례한다.
STAGE_SEC = {"패스1": 3.5, "정련": 12.5, "이음새": 4.4, "후처리": 1.0, "비교": 2.0}
MASK_BYTES_PER_MP = 2000          # 실측 1.5KB/MP(하늘). 대상이 잘게 흩어지면 커진다.


def mask_bytes_per_frame(sizes):
    return int(sum(w * h for w, h in sizes) / max(1, len(sizes)) / 1e6 * MASK_BYTES_PER_MP)


def plan_report(files, dst, stage):
    """시작 전에 예상 소요·용량·여유를 낸다. 다 돌리고 나서 디스크가 찼다는 걸 알면 늦다."""
    if not files:
        return
    sample = files[:: max(1, len(files) // 8)][:8]
    sizes = []
    for f in sample:
        try:
            sizes.append(image_size(f))
        except Exception:
            pass
    if not sizes:
        return
    mp = sum(w * h for w, h in sizes) / len(sizes) / 1e6
    sec = STAGE_SEC.get(stage, 3.0) * mp / 30.2 * len(files)
    need = mask_bytes_per_frame(sizes) * len(files)
    free = free_space(os.path.abspath(dst))
    line = f"  예상   소요 {fmt_dur(sec)}  용량 {fmt_bytes(need)}"
    if free >= 0:
        line += f"  (여유 {fmt_bytes(free)})"
    print(line, flush=True)
    if 0 <= free < need * 1.2:
        print("  ※ 여유 공간이 빠듯하다 — 중간에 멈춘다.", flush=True)


class Progress:
    """경과·속도·남은시간·완료예정시각을 한 줄로."""

    def __init__(self, total, every=25):
        import time as _t
        self.total, self.every, self.t0, self.n = total, every, _t.time(), 0

    def tick(self, k=1):
        import time as _t
        self.n += k
        if self.n % self.every:
            return
        el = _t.time() - self.t0
        rate = el / max(1, self.n)
        left = (self.total - self.n) * rate
        eta = _t.strftime("%H:%M", _t.localtime(_t.time() + left))
        print(f"  {self.n:,}/{self.total:,}  {rate:.1f}s/장  경과 {fmt_dur(el)}  "
              f"남은 {fmt_dur(left)}  완료예정 {eta}", flush=True)

    def finish(self, dst=None, note=""):
        import time as _t
        el = _t.time() - self.t0
        rate = el / max(1, self.n)
        line = f"완료 {self.n:,}장  소요 {fmt_dur(el)}  ({rate:.1f}s/장)"
        if note:
            line += f"  {note}"
        print(line, flush=True)
        if dst:
            ap = os.path.abspath(dst)
            n, b = dir_stats(ap)
            print(f"  출력 {ap}   {n:,}개  {fmt_bytes(b)}", flush=True)


# ---------------------------------------------------------------- 타일

def tile_boxes(W, H, cols, rows, ov=0.25, wrap=True):
    """겹치는 타일의 (x인덱스배열, y0, y1).

    wrap=True 는 가로를 360°로 감는다 — ERP 전용이다. 일반 프레임에 켜면
    화면 왼쪽 끝을 오른쪽 끝과 같은 장면으로 착각한다.
    계약: 모든 화소가 최소 한 타일에 덮이고, 겹침이 존재한다.
    """
    cols, rows = max(1, min(cols, W)), max(1, min(rows, H))
    tw, th = max(1, W // cols), max(1, H // rows)
    px, py = max(1, int(tw * ov)), max(1, int(th * ov))
    out = []
    for r in range(rows):
        y0 = max(0, r * th - py)
        # ★마지막 행/열은 끝까지 늘린다. W//cols 의 나머지가 안 덮이면 그 화소는
        #   가중치 0 이 되어 조용히 '비대상'으로 굳는다(실측 401x121 에서 521px).
        y1 = H if r == rows - 1 else min(H, (r + 1) * th + py)
        for c in range(cols):
            x_hi = (W + px) if (c == cols - 1 and not wrap) else ((c + 1) * tw + px)
            xs = np.arange(c * tw - px, x_hi)
            xs = xs % W if wrap else xs[(xs >= 0) & (xs < W)]
            if len(xs) == 0:
                continue
            out.append((xs, y0, y1))
    return out


# ---------------------------------------------------------------- 추론

class SkyModel:
    """ADE20K 시맨틱 분할로 '하늘다움' 점수를 낸다.

    ★이진 마스크가 아니라 점수를 원본 크기로 확대한 뒤 문턱을 넘긴다.
      이진 마스크를 확대하면 경계가 계단이 된다(실측 14px 단위).
    ★150채널을 통째로 확대하면 7776x3888 에서 21GB 다. argmax 는
      '하늘 로짓 - 나머지 최대' 의 부호와 같으므로 그 1채널만 확대한다(144MB).
    """

    def __init__(self, name="oneformer", device="cuda"):
        import torch
        self.name, self.device, self.torch = name, device, torch
        mid = MODELS[name]
        if name == "oneformer":
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
            self.proc = OneFormerProcessor.from_pretrained(mid)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(mid).to(device).eval()
        elif name == "mask2former":
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            self.proc = AutoImageProcessor.from_pretrained(mid)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(mid).to(device).eval()
        else:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            self.proc = SegformerImageProcessor.from_pretrained(mid)
            self.model = SegformerForSemanticSegmentation.from_pretrained(mid).to(device).eval()

    def score(self, pil, classes=(SKY_CLASS,), infer_width=1024):
        """(1,1,h,w) '대상다움' = 대상 클래스 최대 − 나머지 최대.
        argmax 가 대상에 속하는지와 부호가 같아, 여러 클래스를 한 대상으로 묶을 수 있다.
        프로세서가 자체 크기로 다시 줄이는 모델은 infer_width 를 무시한다
        (mask2former 384, oneformer 640)."""
        torch = self.torch
        if self.name == "segformer":
            ih = max(1, round(pil.size[1] * infer_width / pil.size[0]))
            from PIL import Image
            inp = self.proc(images=pil.resize((infer_width, ih), Image.BILINEAR),
                            return_tensors="pt", do_resize=False).to(self.device)
            with torch.no_grad():
                lg = self.model(**inp).logits
        else:
            kw = {"task_inputs": ["semantic"]} if self.name == "oneformer" else {}
            inp = self.proc(images=pil, return_tensors="pt", **kw).to(self.device)
            with torch.no_grad():
                o = self.model(**inp)
            cls = o.class_queries_logits.softmax(-1)[..., :-1]
            lg = torch.einsum("bqc,bqhw->bchw", cls, o.masks_queries_logits.sigmoid())
        idx = list(classes)
        keep = torch.zeros(lg.shape[1], dtype=torch.bool, device=lg.device)
        keep[idx] = True
        return lg[:, keep].max(1, keepdim=True).values - lg[:, ~keep].max(1, keepdim=True).values

    def target(self, arr, classes=(SKY_CLASS,), cols=4, rows=2, overlap=0.25,
               wrap=True, infer_width=1024):
        """HxWx3 uint8 RGB -> 대상 불리언. 타일을 hann 가중으로 합친다."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        H, W = arr.shape[:2]
        acc = torch.zeros((H, W), device=self.device)
        wgt = torch.zeros_like(acc)
        for xs, y0, y1 in tile_boxes(W, H, cols, rows, overlap, wrap):
            t = arr[y0:y1][:, xs]
            d = self.score(Image.fromarray(t), classes, infer_width)
            up = F.interpolate(d, size=t.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            w2 = (torch.hann_window(t.shape[0], device=self.device).clamp(min=1e-3)[:, None]
                  * torch.hann_window(t.shape[1], device=self.device).clamp(min=1e-3)[None, :])
            xt = torch.as_tensor(xs, device=self.device)
            acc[y0:y1].index_add_(1, xt, up * w2)
            wgt[y0:y1].index_add_(1, xt, w2)
        return ((acc / wgt.clamp(min=1e-6)) > 0).cpu().numpy()


# ---------------------------------------------------------------- 지표

def edge_distance(img_gray):
    """원본 에지까지의 거리맵. 경계 정확도의 대리 지표에 쓴다."""
    import cv2
    e = cv2.Canny(cv2.GaussianBlur(img_gray, (3, 3), 0), 60, 160)
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3)


def boundary_metrics(sky, dist):
    """(중앙거리, 3px 이내 비율). ★이 지표는 '어떤' 에지인지 모른다 —
    아무 질감선에 붙어도 점수가 나온다. 반드시 확대 이미지와 함께 판정하라."""
    import cv2
    m = np.where(sky, 0, 255).astype(np.uint8)
    b = cv2.Canny(m, 50, 150) > 0
    if b.sum() < 10:
        return float("nan"), float("nan")
    d = dist[b]
    return float(np.median(d)), float(100 * (d <= 3).mean())


def island_stats(sky, min_area=2000):
    """(하늘 속 유지조각 수, 유지영역 속 하늘구멍 수). 가장 큰 성분은 제외."""
    import cv2
    out = []
    for target in (False, True):          # False=유지영역 성분(=하늘속 섬), True=하늘 성분(=구멍)
        n, _, st, _ = cv2.connectedComponentsWithStats((sky == target).astype(np.uint8), 8)
        a = st[1:, cv2.CC_STAT_AREA]
        out.append(sum(1 for i in range(len(a)) if i != int(np.argmax(a)) and a[i] >= min_area)
                   if len(a) else 0)
    return tuple(out)


# ---------------------------------------------------------------- 오버레이

def overlay(img_rgb, sky, label=None):
    """원본 위에 하늘을 칠하고 경계선을 얹는다. 모든 비교 그림이 같은 규약을 쓴다."""
    import cv2
    c = img_rgb.copy()
    m = np.where(sky, 0, 255).astype(np.uint8)
    c[sky] = (OV_ALPHA * c[sky] + (1 - OV_ALPHA) * np.array(OV_COLOR)).astype(np.uint8)
    c[cv2.Canny(m, 50, 150) > 0] = OV_EDGE
    if label:
        cv2.putText(c, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(c, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return c


def crop_most_different(masks, W, H, w=1500, h=400):
    """여러 마스크가 가장 많이 갈리는 구간을 찾는다.
    설정 비교는 여기를 봐야 한다 — 무작위 위치로는 0.1% 차이를 놓친다."""
    if len(masks) < 2:
        return crop_most_detailed(masks[0], W, H, w, h)
    d = np.zeros(masks[0].shape, bool)
    for i in range(1, len(masks)):
        d |= (masks[0] != masks[i])
    return _peak_window(d, W, H, w, h)


def crop_most_detailed(sky, W, H, w=1500, h=400):
    """스카이라인 굴곡이 가장 큰 구간. 단일 마스크를 볼 때 쓴다."""
    b = np.array([np.where(sky[:, x])[0].max() if sky[:, x].any() else -1
                  for x in range(0, W, 8)])
    v = b >= 0
    if v.sum() < 10:
        return max(0, W // 2 - w // 2), max(0, H // 2 - h // 2)
    step = max(4, w // 8)
    var = [b[i:i + step][v[i:i + step]].std() if v[i:i + step].sum() > 2 else 0
           for i in range(max(1, len(b) - step))]
    xi = int(np.argmax(var)) * 8
    seg = b[int(np.argmax(var)):int(np.argmax(var)) + step]
    ym = int(seg[seg >= 0].mean()) if (seg >= 0).any() else H // 2
    return max(0, min(xi - w // 2, W - w)), max(0, min(ym - h // 2, H - h))


def _peak_window(mask_bool, W, H, w, h):
    col = mask_bool.sum(0).astype(float)
    xc = int(np.argmax(np.convolve(col, np.ones(w), "same")))
    x0 = max(0, min(xc - w // 2, W - w))
    row = mask_bool[:, x0:x0 + w].sum(1).astype(float)
    yc = int(np.argmax(np.convolve(row, np.ones(h), "same")))
    return x0, max(0, min(yc - h // 2, H - h))


def contact_sheet(img_rgb, entries, box, zoom=1, gap=8, vertical=False):
    """entries: [(라벨, 하늘불리언 또는 None)] — None 이면 원본만.
    box: (x0, y0, w, h). 반환 RGB 배열."""
    import cv2
    x0, y0, w, h = box
    cells = []
    for label, sky in entries:
        c = img_rgb[y0:y0 + h, x0:x0 + w]
        c = overlay(c, sky[y0:y0 + h, x0:x0 + w], label) if sky is not None else c.copy()
        if sky is None and label:
            cv2.putText(c, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(c, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if zoom != 1:
            c = cv2.resize(c, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST)
        cells.append(c)
    ax = 0 if vertical else 1
    pad_shape = ((gap, cells[0].shape[1], 3) if vertical else (cells[0].shape[0], gap, 3))
    pad = np.full(pad_shape, 255, np.uint8)
    parts = []
    for i, c in enumerate(cells):
        parts.append(c)
        if i != len(cells) - 1:
            parts.append(pad)
    return np.concatenate(parts, axis=ax)


def demo():
    cov = np.zeros((120, 400), int)
    for xs, y0, y1 in tile_boxes(400, 120, 8, 4, 0.25, True):
        cov[y0:y1][:, xs] += 1
    assert cov.min() >= 1 and cov.max() > 1, "타일 커버리지/겹침 계약 위반"
    xs0 = tile_boxes(400, 120, 4, 2, 0.25, True)[0][0]
    assert xs0.min() == 0 and xs0.max() == 399, "wrap 이 감싸지 않았다"
    xs1 = tile_boxes(400, 120, 4, 2, 0.25, False)[0][0]
    assert xs1.min() == 0 and xs1.max() < 400, "no-wrap 인데 감쌌다"
    # 퇴화 케이스: 나누어떨어지지 않는 폭, 겹침 0, 격자가 이미지보다 큰 경우
    for W, H, c_, r_, o_, wr in [(401, 121, 4, 2, 0.0, True), (400, 121, 4, 2, 0.0, False),
                                 (400, 10, 4, 4, 0.25, True), (4, 3, 8, 8, 0.25, False),
                                 (7, 5, 3, 2, 0.1, True)]:
        cov = np.zeros((H, W), int)
        for xs, y0, y1 in tile_boxes(W, H, c_, r_, o_, wr):
            cov[y0:y1][:, xs] += 1
        assert cov.min() >= 1, f"덮이지 않은 화소 {int((cov==0).sum())}개 @ {W}x{H} {c_}x{r_} ov={o_}"

    s = np.zeros((60, 60), bool); s[:30] = True
    s[40:44, 40:44] = True                     # 하늘 속 유지영역 밖의 작은 하늘 섬
    s[5:9, 5:9] = False                        # 하늘 속 유지조각
    isl, hol = island_stats(s, min_area=4)
    assert isl >= 1 and hol >= 1, f"섬/구멍 집계 오류 {isl},{hol}"

    img = np.zeros((60, 60, 3), np.uint8)
    o = overlay(img, s)
    assert o[0, 0].tolist() != [0, 0, 0], "하늘이 칠해지지 않았다"
    assert o[50, 0].tolist() == [0, 0, 0], "유지영역이 칠해졌다"
    sheet = contact_sheet(img, [("a", None), ("b", s)], (0, 0, 60, 60))
    assert sheet.shape[1] == 60 * 2 + 8, "콘택트시트 폭 계약 위반"
    assert fmt_bytes(0) == "0B" and fmt_bytes(1536) == "1.5KB"
    assert fmt_bytes(3 * 1024 ** 3) == "3.0GB"
    assert fmt_dur(30) == "30초" and fmt_dur(90) == "1.5분" and fmt_dur(7200) == "2.0시간"
    assert mask_bytes_per_frame([(7776, 3888)]) == 60466
    assert free_space(os.path.dirname(os.path.abspath(__file__))) > 0
    assert dir_stats(os.path.join(os.path.abspath(__file__), "없는폴더")) == (0, 0)

    print("common demo OK")


if __name__ == "__main__":
    demo()
