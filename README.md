# semantic-mask

이미지 폴더에서 **하늘·수면·사람·차량** 같은 대상의 마스크를 대량으로 만든다.
3D Gaussian Splatting, COLMAP, NeRF 학습에서 특정 클래스를 손실에서 빼는 용도다.

Claude Code 스킬로 만들어졌다. 스크립트만 따로 써도 된다.

**마스크 규약: `0` = 대상(제외), `255` = 유지.**

---

## 결과

항공 360° 파노라마(7776×3888) 875장에서 하늘을 제거한 사례다.

![결과](docs/images/01-result.jpg)

무작위 4프레임. 각 쌍의 위가 원본, 아래가 마스크 오버레이(빨강 = 제외, 노랑 = 경계선).

### 경계 정련

![정련](docs/images/02-refine.jpg)

원본 · 분할 결과 · 정련 후. 분할만으로는 경계가 무디다.
정련을 거치면 경계가 실제 구조물 모서리에 붙는다 — 3px 이내 비율 **68% → 98%**.

### 과노출 태양

![태양](docs/images/03-sun.jpg)

태양은 하얗게 날아가 하늘로 판정되지 않는다. 875장 중 139장에서 나타났다.
후처리에서 "하늘 한복판에 떠 있는 조각"으로 걸러 제거한다.

### 모델에 따라 실패하는 곳이 다르다

![구름](docs/images/04-cloud.jpg)

같은 프레임, 다른 모델. 가운데는 엷은 구름을 하늘로 보지 않아 큰 구멍을 남긴다.
오른쪽(OneFormer)은 정상이다. 모델마다 무너지는 지점이 달라서, 몇 장만 보고 고르면 위험하다.

---

## 마스킹 대상

| 대상 | 내용 |
|---|---|
| `sky` | 하늘 |
| `water` | 물·바다·강·호수·수영장 |
| `people` | 사람 |
| `vehicles` | 차·버스·트럭·오토바이·자전거 |
| `movers` | 사람 + 차량 (이동체 제거용) |
| `vegetation` | 나무·풀 |
| `ground` | 도로·땅·모래 |
| `building` | 건물 |
| `c:<번호>` | ADE20K 150클래스 직접 지정 |

대상마다 정책이 다르다. 예를 들어 하늘은 "떠 있는 조각은 오검출"이지만,
사람·차는 그 자체가 떠 있는 조각이라 같은 규칙을 쓰면 대상을 지운다.
프리셋이 이 차이를 처리한다.

---

## 쓰는 법

```bash
PY=python                    # ← CUDA 가 있는 인터프리터
S=scripts

$PY $S/check_env.py                              # 환경 검사 (설치는 제안만)
$PY $S/targets.py                                # 대상 목록
$PY $S/analyze.py  images/ --target sky          # ERP/일반 판별 + 설정 추천
$PY $S/segment.py  images/ coarse/ --target sky  # 분할
$PY $S/refine.py   images/ coarse/ fine/         # 경계 정련
$PY $S/post.py     coarse/ fine/ final/ --target sky
$PY $S/verify.py   images/ final/ --target sky   # 검증
```

360° 파노라마는 `seam.py` 한 단계가 더 붙는다. `analyze.py` 가 알려준다.

모든 단계가 **절대경로 · 용량 · 예상 시간**을 함께 낸다. `analyze.py` 는 착수 전에
소요 시간과 필요한 디스크 용량을 미리 계산하고, 여유 공간이 모자라면 경고한다.

```
[소스 분석]
  이미지    D:\DJI_001\data\ERP   (875개, 13.3GB)
  875장 전량: 패스1 51.1분, 정련 3.0시간, 이음새 1.1시간(ERP만), 후처리 14.6분
용량 추정   마스크 1장 약 59.0KB  ->  단계당 50.5MB
  중간 산출물 4단계 유지 시 총 201.8MB  (출력 예정 드라이브 여유 190.3GB)
```

진행 중에는 남은 시간과 완료예정시각이, 끝나면 출력 절대경로와 산출 용량이 찍힌다.

설치:

```bash
pip install transformers opencv-python-headless pillow
pip install --no-deps segmentation-refinement     # 경계 정련용
```

PyTorch 는 GPU 에 맞는 빌드를 직접 설치한다. 12GB 카드에서 검증했다.

---

## 360° 파노라마 지원

등장방형(2:1) 파노라마는 좌우 끝이 이어진다. 두 가지를 처리한다.

- **타일링이 가로를 감는다.** 일반 프레임에 이걸 켜면 화면 좌우 끝을 같은 장면으로 착각한다. 종횡비로 자동 판정한다.
- **정련 후 이음새를 복구한다.** 정련기는 파노라마를 모르고 좌우 끝을 따로 다듬어 어긋난다 — 실측에서 최대 13% 불일치가 났다.

---

## 알려진 한계

- **유리 커튼월** — 하늘을 반사하는 면을 정련이 파먹는다. 마스크 연산으로는 못 고친다. 정련 전 마스크를 "넉넉본"으로 따로 내는 편이 안전하다.
- **흐린 원경** — 대기 원근으로 대비가 사라진 스카이라인은 뭉개진다.
- **작은 객체(사람·차)** — 프리셋은 준비돼 있으나 대규모 실측은 하늘만 했다.

수치와 근거는 [`REFERENCE.md`](REFERENCE.md) 에 있다.

---

## 원천 소스

**이 저장소의 모델과 알고리즘은 전부 아래 연구자들의 것이다.**
여기서 만든 것은 대량 처리 파이프라인, 파노라마 대응, 검증 절차뿐이다.

| 구성요소 | 출처 | 라이선스 |
|---|---|---|
| **OneFormer** (기본 분할 모델) | [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer) · [모델](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) | MIT |
| **CascadePSP** (경계 정련) | [hkchengrex/CascadePSP](https://github.com/hkchengrex/CascadePSP) | MIT |
| **Mask2Former** (선택) | [모델](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic) | 모델 카드 확인 |
| **SegFormer** (선택) | [모델](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512) | 모델 카드 확인 |
| **ADE20K** (클래스 정의) | [MIT Scene Parsing](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | 데이터셋 조건 확인 |
| **transformers** | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |

비교 과정에서 함께 시험한 것들(이 저장소에 포함되지 않음):
[SegRefiner](https://github.com/MengyuWang826/SegRefiner) (Apache-2.0),
[ViTMatte](https://huggingface.co/hustvl/vitmatte-base-composition-1k) (Apache-2.0),
[SAM 3](https://huggingface.co/facebook/sam3),
[Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf).

선택 모델 중 Mask2Former·SegFormer 체크포인트는 라이선스가 `other` 로 표기돼 있다.
상업적 이용 전에 각 모델 카드를 직접 확인하라.

이 저장소의 코드는 [MIT](LICENSE) 다.
