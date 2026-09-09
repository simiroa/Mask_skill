# -*- coding: utf-8 -*-
"""마스킹 대상 프리셋 — ADE20K 150클래스 기준.

★대상마다 정책이 다르다. 하늘에서 얻은 규칙을 사람·차에 그대로 쓰면 정반대로 작동한다.

  islands="remove" : 대상 영역 안에 떠 있는 '비대상 조각'을 지운다.
      하늘에서만 성립한다 — 항공영상의 지상 구조물은 전부 스카이라인으로 이어지므로
      하늘 한복판에 떠 있을 것이 원리적으로 없다(태양·글레어·새뿐). 실측 139프레임 제거.
  islands="keep"   : 사람·차·배는 그 자체가 고립된 섬이다. 지우면 대상을 지운다.

  tile 큰 값(4x2)  : 넓은 영역(하늘·수면·식생)은 맥락이 필요하다. 잘게 쪼개면 무너진다
      (실측: 8x4 에서 하늘 42%→32%, 세 모델 모두 재현).
  tile 작은 값(8x4): 작은 객체(사람·차)는 반대로 해상도가 필요하다. 큰 타일에서는 놓친다.
      ※ 이 역전은 하늘 실측의 논리적 귀결이며, 작은 객체로는 아직 미검증이다.
"""

N_CLASSES = 150          # ADE20K. HF 규약의 0-기반 인덱스다(하늘=2, 1-기반 표기와 다름)

TARGETS = {
    "sky": {
        "classes": [2],
        "label": "하늘",
        "tile": (4, 2),
        "islands": "remove",
        "island_thr": 100000,     # 태양 실측 최대 39,365px
        "hole_thr": 2000,         # 파고라 틈 등 실제 구멍은 보존
        "checks": ["zenith", "seam"],
        "note": "ERP 항공영상에서 가장 검증된 대상. 천정·이음새·태양 검사 포함.",
    },
    "water": {
        "classes": [21, 26, 60, 128, 109, 104],   # water sea river lake pool fountain
        "label": "수면",
        "tile": (4, 2),
        "islands": "keep",        # 수면 위의 배·부표는 실제로 섬이다
        "island_thr": 0,
        "hole_thr": 2000,
        "checks": ["seam"],
        "note": "반사·윤슬에서 판정이 흔들린다. 하늘과 혼동될 수 있으니 표본 확인 필수.",
    },
    "people": {
        "classes": [12],
        "label": "사람",
        "tile": (8, 4),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 0,
        "checks": [],
        "note": "작은 객체. 타일을 잘게 써야 잡힌다. 팽창(dilate)으로 여유를 주는 게 보통.",
    },
    "vehicles": {
        "classes": [20, 80, 83, 102, 116, 127],   # car bus truck van minibike bicycle
        "label": "차량",
        "tile": (8, 4),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 0,
        "checks": [],
        "note": "작은 객체. 이동체 제거용이면 people 과 합쳐 movers 를 쓴다.",
    },
    "movers": {
        "classes": [12, 20, 80, 83, 102, 116, 127],
        "label": "이동체(사람+차량)",
        "tile": (8, 4),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 0,
        "checks": [],
        "note": "3DGS·SfM 에서 움직이는 것을 손실에서 빼는 표준 조합.",
    },
    "vegetation": {
        "classes": [4, 9, 17],                    # tree grass plant
        "label": "식생",
        "tile": (4, 2),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 2000,
        "checks": [],
        "note": "잎 경계가 반투명해 정련 이득이 작다. 마진을 주는 편이 안전하다.",
    },
    "ground": {
        "classes": [6, 13, 46, 29, 52, 54],       # road earth sand field path runway
        "label": "지면",
        "tile": (4, 2),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 2000,
        "checks": [],
        "note": "지면 위 물체가 구멍으로 남는 게 정상이다. hole 문턱을 낮추지 마라.",
    },
    "building": {
        "classes": [1, 25, 48],                   # building house skyscraper
        "label": "건물",
        "tile": (4, 2),
        "islands": "keep",
        "island_thr": 0,
        "hole_thr": 2000,
        "checks": [],
        "note": "유리 커튼월이 하늘을 반사해 판정이 흔들린다(하늘 작업에서 실측).",
    },
}


def resolve(name):
    """프리셋 이름 또는 'c:2,21' 형식의 직접 지정을 해석한다."""
    if name in TARGETS:
        return dict(TARGETS[name], name=name)
    if name.startswith("c:"):
        try:
            ids = [int(x) for x in name[2:].split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit(f"클래스 인덱스는 정수여야 한다: {name}")
        if not ids:
            raise SystemExit("클래스 인덱스가 비었다 (예: c:2 또는 c:12,20)")
        bad = [i for i in ids if not 0 <= i < N_CLASSES]
        if bad:
            raise SystemExit(f"ADE20K 인덱스 범위를 벗어났다(0~{N_CLASSES-1}): {bad}")
        ids = sorted(set(ids))
        return dict(TARGETS["sky"], name=name, classes=ids, label=f"클래스 {ids}",
                    islands="keep", island_thr=0, checks=[],
                    note="직접 지정. 정책은 보수적 기본값(섬 보존)이다.")
    raise SystemExit(f"모르는 대상: {name}\n사용 가능: {', '.join(TARGETS)} 또는 c:<인덱스,...>")


def describe():
    lines = ["사용 가능한 마스킹 대상:"]
    for k, v in TARGETS.items():
        lines.append(f"  {k:11s} {v['label']:16s} 클래스 {v['classes']}  타일 {v['tile'][0]}x{v['tile'][1]}"
                     f"  섬 {v['islands']}")
        lines.append(f"              {v['note']}")
    lines.append(f"  c:<인덱스,...>  ADE20K 인덱스 직접 지정 0~{N_CLASSES-1} (예: c:2 · c:12,20)")
    lines.append("      인덱스 확인:  python -c \"from transformers import AutoConfig; print(AutoConfig.from_pretrained('shi-labs/oneformer_ade20k_swin_large').id2label)\"")
    return "\n".join(lines)


def demo():
    s = resolve("sky")
    assert s["classes"] == [2] and s["islands"] == "remove", "하늘 프리셋 손상"
    p = resolve("people")
    assert p["islands"] == "keep", "사람에서 섬을 지우면 대상을 지운다"
    assert p["tile"] == (8, 4) and s["tile"] == (4, 2), "타일 정책이 대상별로 갈리지 않는다"
    c = resolve("c:20,12,12")
    assert c["classes"] == [12, 20] and c["islands"] == "keep", "직접 지정은 정렬·중복제거·보수적"
    for bad in ("c:200", "c:-1", "c:abc", "c:"):
        try:
            resolve(bad); raise AssertionError(f"잘못된 지정이 통과됐다: {bad}")
        except SystemExit:
            pass
    try:
        resolve("nope"); raise AssertionError("모르는 대상이 통과됐다")
    except SystemExit:
        pass
    print("targets demo OK")


if __name__ == "__main__":
    print(describe())
    demo()
