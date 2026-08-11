"""SSM 학습용 3D 스캔 데이터셋 — STL 파일과 메타데이터를 짝짓는 매니페스트.

STL 자체는 순수 지오메트리라 자기신고 사이즈, 좌우, 카테고리(정상/평발/절단 등) 같은
정보를 전혀 담지 못한다. 이 모듈은 그 메타데이터를 스캔 파일과 짝짓는 CSV 매니페스트의
스키마를 정의하고, 로드 시점에 검증한다(필수 컬럼 누락, 중복 scan_id, 허용 밖 값,
존재하지 않는 STL 경로 등).

매니페스트 CSV 컬럼(순서 무관, 헤더 필수)::

    scan_id, stl_path, side, self_reported_length_mm, category, subject_id, notes

    - scan_id: 스캔 고유 식별자 (중복 불가)
    - stl_path: STL 파일 경로 (매니페스트 CSV 기준 상대경로 또는 절대경로)
    - side: 'left' | 'right'
    - self_reported_length_mm: 자기신고 발 길이(mm). 스캔 자체엔 축척이 없어
      SSM 정규화 시 기준자로 쓰인다 — 실측이 아니라 자기신고이므로 이 값이 절대 크기의
      유일한 오차 원인이 된다(형상 자체는 이 값과 무관하게 정확).
    - category: `ALLOWED_CATEGORIES` 중 하나
    - subject_id: 동일인 여러 스캔을 추적하고 홀드아웃 분리에 쓰기 위한 식별자
    - notes: 자유 텍스트(선택)

사용 예::

    from foot_engine.scan_dataset import load_manifest, category_counts

    records = load_manifest(Path("data/scans/manifest.csv"))
    print(category_counts(records))
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ScanDatasetError

#: 순서대로 시도할 인코딩 — 한국어 Windows 환경에서 CP949(EUC-KR)로 저장된 CSV가
#: 흔해서, UTF-8로 바로 읽으면 `notes` 같은 한글 텍스트가 깨지거나 디코딩 자체가
#: 실패할 수 있다.
_ENCODING_CANDIDATES = ("utf-8-sig", "cp949", "utf-8")

#: 허용되는 카테고리 값 — 오타로 인한 그룹 분산을 막기 위해 고정 집합으로 관리한다.
#: 실제 매니페스트(2026-08-04)에서 관측된 값: normal, flat_foot, cavus(요족),
#: hallux valgus(무지외반). amputee/injury 는 프로젝트 목표상 대상 인구지만 이번
#: 배치엔 없었던 것으로 보여 미리 자리를 남겨둔다. 새 카테고리가 필요하면 여기에
#: 추가하고, 그 그룹의 표본 수가 SSM을 만들기에 충분한지(최소 몇십 건) 확인할 것.
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "normal",
        "flat_foot",
        "cavus",
        "hallux_valgus",
        "amputee_forefoot",
        "amputee_toe",
        "injury",
        "other",
    }
)

ALLOWED_SIDES: frozenset[str] = frozenset({"left", "right"})

_REQUIRED_COLUMNS = (
    "scan_id",
    "stl_path",
    "side",
    "self_reported_length_mm",
    "category",
    "subject_id",
)


@dataclass(slots=True)
class ScanRecord:
    """매니페스트 한 행 = 스캔 하나. `stl_path` 는 로드 시점에 실존 확인까지 마친 상태다.

    `self_reported_length_mm` 은 None 일 수 있다 — SSM 형상(PCA) 학습 자체는
    각 스캔 자신의 실측 길이로 정규화하므로 자기신고값이 없어도 지장이 없고,
    이 값은 최종 절대 크기 보정(기준자)에만 쓰인다 ([[scan-dataset-characteristics]] 참고).
    """

    scan_id: str
    stl_path: Path
    side: str
    category: str
    subject_id: str
    self_reported_length_mm: float | None = None
    notes: str = ""


def _read_text_with_fallback(path: Path) -> str:
    """`_ENCODING_CANDIDATES` 순서로 디코딩을 시도한다.

    Raises:
        ScanDatasetError: 어떤 인코딩으로도 읽지 못한 경우.
    """
    last_error: UnicodeDecodeError | None = None
    for encoding in _ENCODING_CANDIDATES:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ScanDatasetError(
        f"매니페스트 인코딩을 인식하지 못했습니다(시도: {_ENCODING_CANDIDATES}): {path}",
        detail=str(last_error),
    )


def load_manifest(
    csv_path: Path, *, stl_root: Path | None = None, skip_incomplete: bool = False
) -> tuple[list[ScanRecord], list[str]]:
    """매니페스트 CSV 를 읽어 검증된 `ScanRecord` 목록으로 변환한다.

    Args:
        csv_path: 매니페스트 CSV 경로.
        stl_root: `stl_path` 가 상대경로일 때 기준 폴더. 생략하면 CSV 파일 위치 기준.
        skip_incomplete: True 면 필수값이 비어 있거나 STL 파일이 아직 없는 행을
            에러 대신 건너뛰고 경고로만 모은다 — 데이터를 아직 채우는 중일 때 유용하다.

    Returns:
        (검증된 ScanRecord 목록, 경고 메시지 목록). `skip_incomplete=False` 면 경고는
        항상 비어 있다(문제가 있으면 그 자리에서 예외를 던지므로).

    Raises:
        ScanDatasetError: 필수 컬럼 누락, 빈 매니페스트, scan_id 중복, side/category 값이
            허용 범위 밖, STL 파일이 실제로 없는 경우 등 (`skip_incomplete=True` 일 때는
            scan_id 중복처럼 데이터 자체가 모순인 경우만 예외로 남긴다).
    """
    if not csv_path.is_file():
        raise ScanDatasetError(f"매니페스트 파일이 없습니다: {csv_path}")

    root = stl_root or csv_path.parent
    raw_text = _read_text_with_fallback(csv_path)

    with io.StringIO(raw_text) as f:
        reader = csv.DictReader(f)
        missing_cols = set(_REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing_cols:
            raise ScanDatasetError(
                f"매니페스트에 필수 컬럼이 없습니다: {sorted(missing_cols)}",
                detail={"found_columns": reader.fieldnames},
            )
        rows = list(reader)

    if not rows:
        raise ScanDatasetError("매니페스트에 행이 하나도 없습니다.")

    records: list[ScanRecord] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for i, row in enumerate(rows):
        where = f"{i + 2}행"  # 헤더가 1행이므로 데이터는 2행부터

        # scan_id 자체와 중복 여부는 데이터 정합성 문제라 skip_incomplete 여부와 무관하게
        # 항상 예외로 처리한다("아직 안 채웠다"가 아니라 "매니페스트가 모순된다"는 뜻이므로).
        scan_id = (row.get("scan_id") or "").strip()
        if not scan_id:
            raise ScanDatasetError(f"{where}: scan_id 가 비어 있습니다.")
        if scan_id in seen_ids:
            raise ScanDatasetError(f"{where}: scan_id '{scan_id}' 가 중복됩니다.")
        seen_ids.add(scan_id)

        try:
            records.append(_parse_row(row, where=where, scan_id=scan_id, root=root))
        except ScanDatasetError as exc:
            if not skip_incomplete:
                raise
            warnings.append(f"{where} ({scan_id}) 건너뜀: {exc.message}")

    if not records:
        raise ScanDatasetError("검증을 통과한 스캔이 하나도 없습니다.")

    return records, warnings


def _parse_row(row: dict, *, where: str, scan_id: str, root: Path) -> ScanRecord:
    """scan_id 를 제외한 나머지 컬럼을 검증해 `ScanRecord` 하나를 만든다."""
    side = (row.get("side") or "").strip().lower()
    if side not in ALLOWED_SIDES:
        raise ScanDatasetError(
            f"{where} ({scan_id}): side '{side}' 는 {sorted(ALLOWED_SIDES)} 중 하나여야 합니다."
        )

    # "hallux valgus" 처럼 사람이 공백으로 적는 경우가 흔해 언더스코어로 정규화한다.
    category = (row.get("category") or "").strip().lower().replace(" ", "_")
    if category not in ALLOWED_CATEGORIES:
        raise ScanDatasetError(
            f"{where} ({scan_id}): category '{category}' 는 "
            f"{sorted(ALLOWED_CATEGORIES)} 중 하나여야 합니다."
        )

    # 자기신고 길이는 선택 항목이다 — 없어도 SSM 형상 학습엔 지장 없고,
    # 최종 절대 크기 보정(기준자) 단계에서만 그 스캔을 기준자로 못 쓸 뿐이다.
    raw_length = (row.get("self_reported_length_mm") or "").strip()
    length_mm: float | None = None
    if raw_length:
        try:
            length_mm = float(raw_length)
        except ValueError as exc:
            raise ScanDatasetError(
                f"{where} ({scan_id}): self_reported_length_mm 값이 숫자가 아닙니다: {raw_length!r}"
            ) from exc
        if length_mm <= 0:
            raise ScanDatasetError(
                f"{where} ({scan_id}): self_reported_length_mm 은 0보다 커야 합니다: {length_mm}"
            )

    subject_id = (row.get("subject_id") or "").strip()
    if not subject_id:
        raise ScanDatasetError(f"{where} ({scan_id}): subject_id 가 비어 있습니다.")

    raw_path = (row.get("stl_path") or "").strip()
    if not raw_path:
        raise ScanDatasetError(f"{where} ({scan_id}): stl_path 가 비어 있습니다.")
    stl_path = Path(raw_path)
    if not stl_path.is_absolute():
        stl_path = root / stl_path
    if not stl_path.is_file():
        raise ScanDatasetError(f"{where} ({scan_id}): STL 파일을 찾을 수 없습니다: {stl_path}")

    return ScanRecord(
        scan_id=scan_id,
        stl_path=stl_path,
        side=side,
        category=category,
        subject_id=subject_id,
        self_reported_length_mm=length_mm,
        notes=(row.get("notes") or "").strip(),
    )


def category_counts(records: list[ScanRecord]) -> dict[str, int]:
    """카테고리별 개수 — 표본이 너무 적은 그룹(이상치 등)을 조기에 발견하기 위함."""
    counts: dict[str, int] = {c: 0 for c in ALLOWED_CATEGORIES}
    for r in records:
        counts[r.category] += 1
    return counts


def side_counts(records: list[ScanRecord]) -> dict[str, int]:
    """좌/우 개수 — 한쪽으로 크게 치우쳐 있는지 확인용."""
    counts: dict[str, int] = {s: 0 for s in ALLOWED_SIDES}
    for r in records:
        counts[r.side] += 1
    return counts
