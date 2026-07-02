# Picture-detector → ParseBench layout 통합 (설계)

우리 picture-detect 엔진 결과를 ParseBench layout의 **11개 클래스** 예측으로 정합(integrate)하는
규칙을 정의한다. 정합 규칙을 **먼저 확정**하고, element_rule(VG) 채점은 **그 다음**에 한다.
(통합 없이 점수 변화를 예단하지 않는다.)

## 전제

- **우리 엔진** = Picture 후보 박스를 내는 lane들의 결합:
  `baseline-picture(pymupdf4llm) + vector-merge + raster-lane + icon-lane`.
  엔진은 **Picture 후보만** 만든다. 10개 non-Picture 클래스는 만들지 않는다.
  - `raster-lane`: 임베디드 raster 이미지. 배치 rect를 렌더 content-bbox로 조건부 트림(crop
    복구), soft-mask+동일 xref ≥3회 반복 배치는 장식 backdrop으로 drop.
  - `icon-lane`: 아이콘 크기의 **filled 벡터 드로잉 + 작은 raster 이미지**(아이콘은 벡터일
    수도 raster일 수도 있음). table/chart 영역 내부는 제외, nested dedup.
- **11개 클래스** (Core11): Caption, Footnote, Formula, List-item, Page-footer, Page-header,
  **Picture**, Section-header, Table, Text, Title. → Picture 1개 + non-Picture 10개.
- **채점 지표** = `layout_element_rule_pass_rate` (= 리더보드 Visual_Grounding). content 없는
  Picture(전체 Picture GT의 71%, 아이콘 대부분)는 **localization + classification(Picture)만**
  통과하면 되고 attribution은 skip.
- **지금까지 실제 element_rule로 확정된 값은 baseline뿐**: 56.26 (avg) / 57.20 (micro),
  edge_threshold=0.75 시 57.48 / 59.50. lane 실험의 recall/precision은 **자체 대리지표(Picture
  클래스·localization만)** 이며 실제 점수가 아니다.

## 최종 예측 구성

```
최종 layout 예측 =
   (baseline provider의 10개 non-Picture 클래스 박스, 그대로 유지)
 + (우리 엔진 Picture 박스, 아래 규칙 A·B로 정리)
```

우리 엔진은 non-Picture 10개 클래스를 건드리지 않는다. 오직 Picture 예측만 정합한다.

---

## 규칙 A — 우리 엔진 결과 vs **Table 클래스만**

**결정됨 (실험/사례로 확정):** 규칙 A는 10개 non-Picture 클래스 전체가 아니라 **table
클래스에만** 적용한다.

- **Table은 composite 클래스** — 내부의 text·숫자·아이콘을 자기가 소유한다. 따라서 후보가
  table 안에 있으면 그건 table 소유 → drop.
- **나머지 클래스(text/section-header/page-header/footer/list-item/title)는 아이콘·picture와
  공존한다.** text 박스 안/옆의 아이콘도 엄연히 Picture다. 그래서 이들과 겹친다고 drop하면
  안 된다. (근거 사례: `Apple_..._p16` 우측 컬럼 아이콘 4개, `2023-Sustainability_p12` SDG
  첫 아이콘 — to_json "text" 박스가 아이콘을 덮어 잘못 drop되던 버그.)

| 상황 | 처리 |
|---|---|
| **table에 ≥70% 포함** | **무시(제거).** table 소유. |
| **table 경계에 걸침 / table과 안 겹침** | **Picture로 유지** |
| **다른 non-Picture 클래스와 겹침** | **무관 — 유지** (공존) |

이로써 규칙 A는 사실상 **"table 안이면 빼고, 아니면 아이콘/picture로 살린다"**는 icon/table
경계 규정이 된다. 크기·lane 기반 예외는 불필요(table-only가 자연히 아이콘을 살림).

---

## 규칙 B — 우리 엔진 결과 vs (기존) Picture 영역

우리 엔진 Picture 후보 박스를, Picture 영역(baseline picture 또는 다른 lane 결과)과 비교한다.

| 상황 | 처리 |
|---|---|
| **기존 Picture 영역에 완전히 포함** | **[핵심 쟁점]** — 병합? 분할? 무엇을 남길지 |
| **기존 Picture 경계에 걸침 (부분 겹침)** | **[핵심 쟁점]** — 병합? 분할? |
| **아예 분리됨 (안 겹침)** | **Picture로 추가** |

**핵심 쟁점 (완전 포함 / 경계 걸침):**
- 같은 Picture 영역을 여러 lane이 서로 다른 granularity로 잡을 때(예: vmerge=큰 병합 영역,
  icon=개별 아이콘, baseline=중간), **무엇을 최종 Picture로 남길지**를 정해야 한다.
- 옵션 방향: 더 tight/작은 단위 우선 vs 큰 병합 우선 vs GT granularity에 맞춘 규칙.
- 아직 미결. (앞선 실험에서 병합·분할·gap 조정이 페이지 유형마다 상충함을 확인 — pdf_0a9304
  같은 규칙적 아이콘 grid는 분할, BBRI 같은 multi-part 아이콘은 병합이 필요.)

---

## 확정된 결정 목록 (실험으로 확정)

1. **규칙 A — table 클래스에만 적용.** table 안(≥70%)이면 drop, 아니면 유지. 다른 클래스와의
   겹침·경계는 무관. **추가: 페이지 50% 초과 "table"은 오검출로 보고 무시** (Ford p16: 63%
   가짜 table이 Picture GT 10개 전멸시키던 케이스).
2. **규칙 B — scale-aware 포함 dedup.** 작은 후보가 큰 kept 박스에 ≥60% 포함되면 제거하되,
   **kept가 후보의 10배 초과 크기면 흡수 금지** — granularity가 다르면 중복이 아님 (풀페이지
   baseline 박스가 아이콘을 전부 삼키고 자신은 pred-side gate에 걸려 아무도 못 맞추던 Bursa
   p11 케이스). lane 결합은 union 후 이 dedup.
3. **content**: 통합 picture 박스에 박스 내부 텍스트 span을 `content.text`로 채움 (attribution
   회수, +0.66). reading order −3.4는 리더보드 비반영이라 수용 — 필요시
   `build_intg_predictions.py`에서 content off로 전환 가능.

## 최종 결과 (공식 element_rule 채점)

| 지표 | baseline v4 | 통합(v6) | Δ |
|---|---|---|---|
| **element_rule avg (= 리더보드 VG)** | 56.26 | **59.05** | **+2.79** |
| element_rule micro | 57.20 | 60.76 | +3.56 |
| Picture GT 커버리지(proxy) | 64.3% | 77.1% | +12.8pp |

구현·재현 방법: `../picture-detection-bench/README.md` (별도 저장소 `picture-detection-bench/`).

## 알려진 gap (미해결)
- **under/over-merge 잔여** (~180 GT): 병합 granularity 불일치, 분할 로직 필요.
- **struct 내부 드로잉** (~100 GT): text 블록 안으로 필터된 vector 그림.
- **얇은 라인아트(stroke) 아이콘**: 예 `orbia_p19` — fill 없는 outline이라 icon-lane(fill
  기반)·raster 어느 쪽도 못 잡음.
- **풀페이지 raster에 flatten된 그림**: 예 `Lancashire p11` — 그림이 전면 배경 이미지 안에
  구워져 있어 metadata로는 원리적으로 불가, 렌더 기반 분할 필요.

## 관련 파일
- 엔진 lane + 통합 + 조립 + 진단 + vg-perf 뷰어: **`picture-detection-bench/` 저장소** (chart/table-detection-bench와 동급 sibling)
- 채점된 예측: `output/pymupdf4llm_layout_v5_intg/`
- baseline 리포트: `output/pymupdf4llm_layout_v4/layout/_evaluation_report.json`
- lane 분류 체계: `docs/picture_classification.md`
