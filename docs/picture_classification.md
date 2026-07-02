## included

icon lane

- 크기가 작고 정방형인 vector clusters, raster images

raster lane

- 일반 raster image (not iconic)
- clip area, z-index로 화면에 표출되는 부분만 인식
- 4llm이 아니라 pymupdf image로 꺼내와야 함

vector cluster lane

- 인접한 vector들을 병합
- pymupdf layout class의 text like와 중첩이 없거나 아주 작아야 함

logo lane

- glyph 검토 필요

mixed lane

- vector cluster, raster images, text span, glyph가 모두 인접하여 포함된 영역
- pymupdf layout class의 text like와 중첩이 없거나 아주 작아야 함

---

## excluded

- text only box (vector background)

## Findings

- picture component의 추출 순서가 중요할 것으로 사료
