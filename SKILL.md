---
name: semantic-mask
description: Build per-image semantic masks (sky, water, people, vehicles, vegetation, ground, buildings, or any ADE20K class) for photogrammetry, 3D Gaussian Splatting, SfM, or dataset cleaning — segment, refine boundaries, post-process, and verify at scale. Handles equirectangular 360° panoramas (wrap-aware tiling + seam repair) and ordinary frames. Use when the user wants to mask or exclude a class from images, mentions sky masking, sky removal, moving-object masks, water masks, 3DGS/COLMAP/NeRF training masks, ERP panorama masks, or points at an image folder needing masks.
---

# semantic-mask

이미지 폴더에서 특정 대상(하늘·수면·사람·차량 등)의 마스크를 대량 생성한다.
**마스크 규약: 0 = 대상(손실에서 제외), 255 = 유지.**

스크립트는 `scripts/` 에 있다. `common` `targets` `post` `seam` 은 인자 없이 실행하면
자체검사만 돈다. 파이썬은 사용자의 CUDA 환경을 쓴다 — 경로를 먼저 확인하고 `$PY` 에 넣는다.

## 선행 질문 — 시작 전에 반드시 확인

1. **무엇을 마스킹하나?** `python scripts/targets.py` 로 선택지를 보여주고 고르게 한다.
   `sky` · `water` · `people` · `vehicles` · `movers` · `vegetation` · `ground` · `building`
   또는 `c:<ADE20K 인덱스>` 직접 지정. **대상마다 정책이 다르다** — 하늘의 규칙을
   사람·차에 그대로 쓰면 정반대로 작동한다([REFERENCE.md](REFERENCE.md) 정책표).
2. **대상을 빼는가, 남기는가?** 기본은 "대상을 손실에서 제외". 반대면 `--margin` 부호와
   후처리 정책을 뒤집어야 하므로 미리 확인한다.
3. **출력 위치와 이름 규약** — 원본과 같은 폴더에 `<이름>_mask.png` 가 기본.
4. **정밀도와 시간 중 무엇이 우선인가** — 정련 단계(패스2)가 전체 시간의 8할이다.
   빠른 초안이 필요하면 패스1만으로도 쓸 수 있다.

## 워크플로

```bash
PY=python                    # ← CUDA 가 있는 인터프리터로 바꾼다
S=~/.claude/skills/semantic-mask/scripts
IMG=<이미지폴더>

$PY $S/check_env.py                                    # 0. 환경 검사 (설치는 제안만)
$PY $S/targets.py                                      #    마스킹 대상 목록
$PY $S/analyze.py  $IMG --target sky                   # 1. ERP/일반 판별 + 설정·비용
$PY $S/segment.py  $IMG coarse/  --target sky --shard k/4        # 2. 패스1
$PY $S/refine.py   $IMG coarse/ fine/ --L 900                    # 3. 패스2 (경계)
$PY $S/seam.py     $IMG coarse/ fine/ seamed/                    # 4. ERP 전용
$PY $S/post.py     coarse/ seamed/ final/ --target sky --max-grow 5   # 5. 후처리
$PY $S/compare.py  $IMG --masks 성긴=coarse/ 최종=final/ --pick 3      # 6. 눈으로 판정
$PY $S/verify.py   $IMG final/ --target sky                      # 7. 판정 줄 확인
```

ERP 가 아니면 4단계를 건너뛰고 `post.py coarse/ fine/ final/` 로 이어간다.

- **샤딩**: 패스1은 4워커, 패스2는 **단일 워커**가 빠르다(메모리 경합).
- **중간 산출물을 지우지 마라.** `coarse` 가 있으면 후처리·정련 설정을 바꿀 때
  모델 재실행 없이 다시 만든다.
- **대상이 없는 프레임**은 정련을 건너뛴다(`refine.py` 가 자동으로 한다).
  패스1 자체를 건너뛰려면 이전 실행의 `stats.*.jsonl` 에서 목록을 뽑아 `--only` 에 준다:
  `python -c "import json,sys;[print(json.loads(l)['f']) for l in open(sys.argv[1],encoding='utf-8') if json.loads(l)['area']>0]" coarse/stats.0.jsonl > todo.txt`
  나머지는 `segment.py … --fill-empty` 로 255 마스크를 직접 쓴다.

## 반드시 지킬 것

- **수치만으로 판정하지 마라.** 주 지표(3px 이내)는 어떤 에지인지 모른다.
  `compare.py` 가 두 결과의 최대 차이 구간을 자동으로 찾아 확대해준다 — 그 그림으로 정한다.
- **가로 감기는 종횡비로 자동 판정한다**(2:1 ±0.05 만 ERP). 확실히 하려면 `--wrap` / `--no-wrap` 을
  명시하라 — 일반 프레임에 감기를 켜면 화면 좌우 끝을 같은 장면으로 착각한다.
- **단계를 추가하면 앞 단계의 불변식을 다시 검증하라.** `verify.py` 의 판정 줄이 통과할 때까지 확정 금지.
- **막힌 길을 다시 파지 마라** — 타일 세분·입력 해상도 증가·반복 정련·매팅·확산 정련은
  전부 실측으로 기각됐다. 근거와 수치는 [REFERENCE.md](REFERENCE.md).

## 상세

측정된 모델·기법 비교, 대상별 정책표, VRAM 절벽, 알려진 실패 유형은
[REFERENCE.md](REFERENCE.md) 에 있다.
