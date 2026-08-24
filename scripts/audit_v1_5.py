"""v1.5 최종 코드베이스 Audit.

현재 repository를 읽기 전용으로 검사한다.

검사 항목:
1. Python 문법 파싱
2. 사용하지 않는 import 후보
3. 참조되지 않는 production 함수 / 클래스 후보
4. 동일 이름 정의 중복
5. 상수 값 중복
6. requirements와 실제 import 비교
7. 구버전 코드 / 문서 참조
8. 공통 라벨의 재중복 여부
9. 임시 / 백업 파일 잔존 여부
10. Streamlit secret Git 추적 여부
11. 고권한 Supabase secret 문자열 추적 여부

이 스크립트는 파일을 수정하거나 삭제하지 않는다.
출력된 항목은 모두 "검토 후보"이며 자동 삭제 대상이 아니다.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# 검사 범위
# ============================================================

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-v1.4",
    ".venv-v1.5",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "__MACOSX",
    "backups",
    "data",
    "outputs",
}

PYTHON_SCAN_TARGETS = (
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "scripts",
)

PRODUCTION_TARGETS = (
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "src",
)

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


# ============================================================
# package 이름 정규화
# ============================================================

IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
}

LOCAL_IMPORT_ROOTS = {
    "app",
    "main",
    "src",
    "tests",
    "scripts",
}


def normalize_package_name(
    name: str,
) -> str:
    """PyPI 패키지 이름 비교용 정규화."""

    return (
        name.strip()
        .lower()
        .replace("_", "-")
    )


# ============================================================
# 파일 수집
# ============================================================

def is_excluded(
    path: Path,
) -> bool:
    """제외 디렉터리가 경로에 포함되는지 확인한다."""

    try:
        relative = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        return True

    return any(
        part in EXCLUDED_DIRECTORIES
        for part in relative.parts
    )


def iter_python_files(
    targets,
) -> list[Path]:
    """검사할 Python 파일을 반환한다."""

    files: set[Path] = set()

    for target in targets:
        if not target.exists():
            continue

        if target.is_file():
            if (
                target.suffix == ".py"
                and not is_excluded(
                    target
                )
            ):
                files.add(target)

            continue

        for path in target.rglob("*.py"):
            if not is_excluded(path):
                files.add(path)

    return sorted(files)


def iter_text_files() -> list[Path]:
    """구버전 참조 검사에 사용할 텍스트 파일."""

    result: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if (
            not path.is_file()
            or is_excluded(path)
            or path.suffix.lower()
            not in TEXT_SUFFIXES
        ):
            continue

        result.append(path)

    return sorted(result)


def relative(
    path: Path,
) -> str:
    """프로젝트 기준 상대경로."""

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    except ValueError:
        return str(path)


# ============================================================
# AST 파싱
# ============================================================

def parse_python_files(
    paths: list[Path],
) -> tuple[
    dict[Path, ast.Module],
    list[str],
]:
    """Python 파일을 AST로 파싱한다."""

    trees: dict[
        Path,
        ast.Module,
    ] = {}

    errors: list[str] = []

    for path in paths:
        try:
            source = path.read_text(
                encoding="utf-8"
            )

            trees[path] = ast.parse(
                source,
                filename=str(path),
            )

        except Exception as exc:
            errors.append(
                f"{relative(path)}: {exc}"
            )

    return trees, errors


# ============================================================
# 사용하지 않는 import 후보
# ============================================================

def collect_loaded_names(
    tree: ast.AST,
) -> set[str]:
    """실제로 Load context로 사용되는 이름."""

    return {
        node.id
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and isinstance(
                node.ctx,
                ast.Load,
            )
        )
    }


def unused_import_candidates(
    trees: dict[Path, ast.Module],
) -> list[str]:
    """파일별 미사용 import 후보를 찾는다."""

    candidates: list[str] = []

    for path, tree in trees.items():
        loaded_names = (
            collect_loaded_names(
                tree
            )
        )

        for node in ast.walk(tree):

            if isinstance(
                node,
                ast.Import,
            ):
                for alias in node.names:
                    used_name = (
                        alias.asname
                        or alias.name.split(
                            "."
                        )[0]
                    )

                    if (
                        used_name
                        not in loaded_names
                    ):
                        candidates.append(
                            f"{relative(path)}:"
                            f"{node.lineno} "
                            f"import {alias.name}"
                            + (
                                f" as {alias.asname}"
                                if alias.asname
                                else ""
                            )
                        )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):
                if (
                    node.module
                    == "__future__"
                ):
                    continue

                for alias in node.names:
                    if alias.name == "*":
                        continue

                    used_name = (
                        alias.asname
                        or alias.name
                    )

                    if (
                        used_name
                        not in loaded_names
                    ):
                        module = (
                            node.module
                            or ""
                        )

                        candidates.append(
                            f"{relative(path)}:"
                            f"{node.lineno} "
                            f"from {module} "
                            f"import {alias.name}"
                            + (
                                f" as {alias.asname}"
                                if alias.asname
                                else ""
                            )
                        )

    return sorted(candidates)


# ============================================================
# 함수 / 클래스 참조 검사
# ============================================================

def collect_production_definitions(
    trees: dict[Path, ast.Module],
    production_files: set[Path],
) -> list[
    tuple[str, str, int, str]
]:
    """production top-level 함수 / 클래스 정의 수집."""

    definitions = []

    for path, tree in trees.items():
        if path not in production_files:
            continue

        for node in tree.body:
            if isinstance(
                node,
                ast.FunctionDef,
            ):
                definitions.append(
                    (
                        node.name,
                        relative(path),
                        node.lineno,
                        "function",
                    )
                )

            elif isinstance(
                node,
                ast.AsyncFunctionDef,
            ):
                definitions.append(
                    (
                        node.name,
                        relative(path),
                        node.lineno,
                        "async function",
                    )
                )

            elif isinstance(
                node,
                ast.ClassDef,
            ):
                definitions.append(
                    (
                        node.name,
                        relative(path),
                        node.lineno,
                        "class",
                    )
                )

    return definitions


def collect_name_references(
    trees: dict[Path, ast.Module],
) -> dict[str, int]:
    """repository 전체의 이름 참조 횟수."""

    references: dict[
        str,
        int,
    ] = defaultdict(int)

    for tree in trees.values():
        for node in ast.walk(tree):

            if (
                isinstance(
                    node,
                    ast.Name,
                )
                and isinstance(
                    node.ctx,
                    ast.Load,
                )
            ):
                references[
                    node.id
                ] += 1

            elif isinstance(
                node,
                ast.Attribute,
            ):
                references[
                    node.attr
                ] += 1

    return references


def unreferenced_definition_candidates(
    definitions,
    references,
) -> list[str]:
    """한 번도 참조되지 않는 production 정의 후보."""

    candidates = []

    exempt_names = {
        "main",
    }

    for (
        name,
        path,
        line,
        kind,
    ) in definitions:

        if (
            name in exempt_names
            or name.startswith("__")
        ):
            continue

        if references.get(
            name,
            0,
        ) == 0:
            candidates.append(
                f"{path}:{line} "
                f"{kind} {name}"
            )

    return sorted(candidates)


# ============================================================
# 동일 이름 정의 검사
# ============================================================

def duplicate_definition_names(
    definitions,
) -> list[str]:
    """production 여러 파일에 동일한 top-level 이름이 있는지 확인."""

    grouped = defaultdict(list)

    for (
        name,
        path,
        line,
        kind,
    ) in definitions:
        grouped[name].append(
            (
                path,
                line,
                kind,
            )
        )

    result = []

    for name, locations in sorted(
        grouped.items()
    ):
        if len(locations) < 2:
            continue

        formatted = ", ".join(
            f"{path}:{line}"
            for (
                path,
                line,
                _kind,
            ) in locations
        )

        result.append(
            f"{name}: {formatted}"
        )

    return result


# ============================================================
# 상수 중복 검사
# ============================================================

def collect_literal_constants(
    trees: dict[Path, ast.Module],
    production_files: set[Path],
):
    """literal_eval 가능한 대문자 상수를 수집한다."""

    values = []

    for path, tree in trees.items():
        if path not in production_files:
            continue

        for node in tree.body:
            name = None
            value_node = None

            if (
                isinstance(
                    node,
                    ast.Assign,
                )
                and len(
                    node.targets
                ) == 1
                and isinstance(
                    node.targets[0],
                    ast.Name,
                )
            ):
                name = (
                    node.targets[0].id
                )
                value_node = node.value

            elif (
                isinstance(
                    node,
                    ast.AnnAssign,
                )
                and isinstance(
                    node.target,
                    ast.Name,
                )
                and node.value
                is not None
            ):
                name = node.target.id
                value_node = node.value

            if (
                not name
                or not name.isupper()
            ):
                continue

            try:
                value = ast.literal_eval(
                    value_node
                )
            except Exception:
                continue

            values.append(
                (
                    name,
                    relative(path),
                    node.lineno,
                    repr(value),
                )
            )

    return values


def duplicate_constant_values(
    constants,
) -> list[str]:
    """서로 다른 대문자 상수가 같은 literal 값을 갖는 후보."""

    grouped = defaultdict(list)

    for (
        name,
        path,
        line,
        value_repr,
    ) in constants:

        # 너무 흔한 단일 bool / None 등은
        # 중복 후보에서 제외한다.
        if value_repr in {
            "True",
            "False",
            "None",
            "0",
            "1",
        }:
            continue

        grouped[
            value_repr
        ].append(
            (
                name,
                path,
                line,
            )
        )

    result = []

    for locations in grouped.values():
        unique_names = {
            name
            for (
                name,
                _path,
                _line,
            ) in locations
        }

        if (
            len(locations) < 2
            or len(unique_names) < 2
        ):
            continue

        formatted = ", ".join(
            f"{name} ({path}:{line})"
            for (
                name,
                path,
                line,
            ) in locations
        )

        result.append(
            formatted
        )

    return sorted(result)


# ============================================================
# requirements 비교
# ============================================================

def parse_requirement_file(
    path: Path,
    seen: set[Path] | None = None,
) -> set[str]:
    """-r 포함 requirements를 재귀적으로 읽는다."""

    if seen is None:
        seen = set()

    path = path.resolve()

    if (
        path in seen
        or not path.exists()
    ):
        return set()

    seen.add(path)

    packages: set[str] = set()

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
        ):
            continue

        if line.startswith(
            "-r "
        ):
            child = (
                path.parent
                / line[3:].strip()
            )

            packages |= (
                parse_requirement_file(
                    child,
                    seen,
                )
            )

            continue

        if line.startswith(
            "--requirement "
        ):
            child = (
                path.parent
                / line.split(
                    None,
                    1,
                )[1]
            )

            packages |= (
                parse_requirement_file(
                    child,
                    seen,
                )
            )

            continue

        if line.startswith("-"):
            continue

        package_name = re.split(
            r"[\[<>=!~;\s]",
            line,
            maxsplit=1,
        )[0]

        if package_name:
            packages.add(
                normalize_package_name(
                    package_name
                )
            )

    return packages


def collect_import_roots(
    trees: dict[Path, ast.Module],
) -> set[str]:
    """전체 Python 소스에서 import root를 수집한다."""

    roots = set()

    for tree in trees.values():
        for node in ast.walk(tree):

            if isinstance(
                node,
                ast.Import,
            ):
                for alias in node.names:
                    roots.add(
                        alias.name.split(
                            "."
                        )[0]
                    )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):
                if node.module:
                    roots.add(
                        node.module.split(
                            "."
                        )[0]
                    )

    return roots


def third_party_import_packages(
    roots: set[str],
) -> set[str]:
    """stdlib / local import를 제외한 외부 package 이름."""

    stdlib = set(
        getattr(
            sys,
            "stdlib_module_names",
            set(),
        )
    )

    packages = set()

    for root in roots:
        if (
            root in stdlib
            or root in LOCAL_IMPORT_ROOTS
            or root == "__future__"
        ):
            continue

        package = IMPORT_TO_PACKAGE.get(
            root,
            root,
        )

        packages.add(
            normalize_package_name(
                package
            )
        )

    return packages


# ============================================================
# 오래된 참조 검사
# ============================================================

STALE_PATTERNS = {
    "삭제된 report 모듈":
        r"\bsrc\.report\b",

    "구 clean_with_pandas API":
        r"\bclean_with_pandas\b",

    "구 welch_test API":
        r"\bwelch_test\b",

    "구 create_visualizations API":
        r"\bcreate_visualizations\b",

    "구 모델 파일명":
        r"\bincome_pipeline\.joblib\b",

    "구 SERVICE_VERSION 상수":
        r"\bSERVICE_VERSION\b",

    "구 README_APPLY 문서":
        r"\bREADME_APPLY\.md\b",

    "구 v1.3 requirements lock":
        r"\brequirements-v1\.3-lock\.txt\b",

    "polars import":
        r"(?:^|\s)(?:import polars|from polars)",

    "seaborn import":
        r"(?:^|\s)(?:import seaborn|from seaborn)",

    "supabase-py import":
        r"(?:^|\s)(?:import supabase|from supabase)",

    "postgrest import":
        r"(?:^|\s)(?:import postgrest|from postgrest)",
}


def stale_reference_candidates(
    text_files: list[Path],
) -> list[str]:
    """과거 API / 파일 / dependency 참조를 찾는다."""

    result = []

    compiled = {
        label: re.compile(
            pattern,
            flags=re.MULTILINE,
        )
        for label, pattern
        in STALE_PATTERNS.items()
    }

    for path in text_files:
        # Audit 규칙 문자열 자체를 stale reference로 오탐하지 않는다.
        if path.resolve() == Path(__file__).resolve():
            continue

        try:
            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception:
            continue

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            for label, pattern in (
                compiled.items()
            ):
                if pattern.search(line):
                    result.append(
                        f"{relative(path)}:"
                        f"{line_number} "
                        f"[{label}]"
                    )

    return sorted(result)


# ============================================================
# 라벨 중앙화 검사
# ============================================================

def duplicate_label_definitions(
    trees: dict[Path, ast.Module],
) -> list[str]:
    """labels.py 외부의 라벨 상수 재정의를 찾는다."""

    target_names = {
        "VARIABLE_LABELS",
        "CATEGORY_VALUE_LABELS",
        "VARIABLE_TYPE_LABELS",
    }

    result = []

    labels_path = (
        PROJECT_ROOT
        / "src"
        / "labels.py"
    ).resolve()

    for path, tree in trees.items():
        if path.resolve() == labels_path:
            continue

        for node in tree.body:

            assigned_names = []

            if isinstance(
                node,
                ast.Assign,
            ):
                for target in node.targets:
                    if isinstance(
                        target,
                        ast.Name,
                    ):
                        assigned_names.append(
                            target.id
                        )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    assigned_names.append(
                        node.target.id
                    )

            for name in assigned_names:
                if name in target_names:
                    result.append(
                        f"{relative(path)}:"
                        f"{node.lineno} "
                        f"{name}"
                    )

    return sorted(result)


# ============================================================
# 파일 구조 검사
# ============================================================

EXPECTED_FILES = (
    "app.py",
    "main.py",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "src/config.py",
    "src/labels.py",
    "src/inquiry.py",
    "src/ui_common.py",
    "src/ui_inquiry.py",
    "src/ui_association.py",
    "src/ui_prediction.py",
)

SHOULD_BE_ABSENT = (
    "README_APPLY.md",
    "requirements-v1.3-lock.txt",
    "app.before_step5b.py",
    "scripts/refactor_ui_step5b.py",
)


def file_structure_issues() -> list[str]:
    """필수 / 임시 파일 존재 여부를 검사한다."""

    result = []

    for relative_path in EXPECTED_FILES:
        path = (
            PROJECT_ROOT
            / relative_path
        )

        if not path.exists():
            result.append(
                "필수 파일 없음: "
                f"{relative_path}"
            )

    for relative_path in SHOULD_BE_ABSENT:
        path = (
            PROJECT_ROOT
            / relative_path
        )

        if path.exists():
            result.append(
                "정리 후보 파일 존재: "
                f"{relative_path}"
            )

    return result


# ============================================================
# Git / Secret 검사
# ============================================================

def run_git(
    *args: str,
) -> subprocess.CompletedProcess:
    """Git 명령을 안전하게 실행한다."""

    return subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_security_issues() -> list[str]:
    """tracked secret / 고권한 key 패턴을 확인한다."""

    result = []

    tracked_secret = run_git(
        "ls-files",
        ".streamlit/secrets.toml",
    )

    if tracked_secret.stdout.strip():
        result.append(
            ".streamlit/secrets.toml이 "
            "Git에 추적되고 있습니다."
        )

    tracked = run_git(
        "ls-files",
        "-z",
    )

    if tracked.returncode != 0:
        result.append(
            "Git tracked file 검사를 "
            "수행하지 못했습니다."
        )
        return result

    secret_patterns = {
        "Supabase secret key":
            re.compile(
                r"\bsb_secret_[A-Za-z0-9._-]+"
            ),

        "Supabase service role 변수":
            re.compile(
                r"\bSUPABASE_SERVICE_ROLE"
                r"(?:_KEY)?\b"
            ),

        "DB URL 자격증명":
            re.compile(
                r"postgres(?:ql)?://"
                r"[^:\s]+:[^@\s]+@"
            ),
    }

    tracked_paths = [
        item
        for item in tracked.stdout.split(
            "\0"
        )
        if item
    ]

    for relative_path in tracked_paths:
        path = (
            PROJECT_ROOT
            / relative_path
        )

        if (
            not path.exists()
            or not path.is_file()
        ):
            continue

        try:
            if path.stat().st_size > (
                2 * 1024 * 1024
            ):
                continue

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

        except Exception:
            continue

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            for label, pattern in (
                secret_patterns.items()
            ):
                if pattern.search(line):
                    result.append(
                        f"{relative_path}:"
                        f"{line_number} "
                        f"[{label}]"
                    )

    return result


# ============================================================
# 출력 helper
# ============================================================

def print_section(
    title: str,
    items,
    *,
    empty_message: str = "없음",
) -> None:
    """Audit 결과 섹션 출력."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    items = list(items)

    if not items:
        print(empty_message)
        return

    for item in items:
        print(f"- {item}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    python_files = (
        iter_python_files(
            PYTHON_SCAN_TARGETS
        )
    )

    production_files = set(
        iter_python_files(
            PRODUCTION_TARGETS
        )
    )

    text_files = (
        iter_text_files()
    )

    trees, parse_errors = (
        parse_python_files(
            python_files
        )
    )

    print(
        "v1.5 FINAL AUDIT"
    )

    print(
        f"Python files scanned: "
        f"{len(python_files)}"
    )

    print_section(
        "1. Python AST 파싱 오류",
        parse_errors,
    )

    unused_imports = (
        unused_import_candidates(
            trees
        )
    )

    print_section(
        "2. 사용하지 않는 import 후보",
        unused_imports,
        empty_message=(
            "발견되지 않음"
        ),
    )

    definitions = (
        collect_production_definitions(
            trees,
            production_files,
        )
    )

    references = (
        collect_name_references(
            trees
        )
    )

    unreferenced = (
        unreferenced_definition_candidates(
            definitions,
            references,
        )
    )

    print_section(
        "3. 참조되지 않는 production 정의 후보",
        unreferenced,
        empty_message=(
            "발견되지 않음"
        ),
    )

    duplicate_defs = (
        duplicate_definition_names(
            definitions
        )
    )

    print_section(
        "4. 동일 이름의 production 정의",
        duplicate_defs,
        empty_message=(
            "발견되지 않음"
        ),
    )

    constants = (
        collect_literal_constants(
            trees,
            production_files,
        )
    )

    duplicated_constants = (
        duplicate_constant_values(
            constants
        )
    )

    print_section(
        "5. 동일 값을 가진 상수 후보",
        duplicated_constants,
        empty_message=(
            "발견되지 않음"
        ),
    )

    import_roots = (
        collect_import_roots(
            trees
        )
    )

    imported_packages = (
        third_party_import_packages(
            import_roots
        )
    )

    requirements_path = (
        PROJECT_ROOT
        / "requirements-dev.txt"
    )

    if not requirements_path.exists():
        requirements_path = (
            PROJECT_ROOT
            / "requirements.txt"
        )

    declared_packages = (
        parse_requirement_file(
            requirements_path
        )
    )

    missing_requirements = sorted(
        imported_packages
        - declared_packages
    )

    possibly_unused_requirements = sorted(
        declared_packages
        - imported_packages
    )

    print_section(
        "6-A. 코드에서 import하지만 requirements에 없는 package",
        missing_requirements,
        empty_message=(
            "발견되지 않음"
        ),
    )

    print_section(
        "6-B. requirements에 있으나 직접 import되지 않는 package 후보",
        possibly_unused_requirements,
        empty_message=(
            "발견되지 않음"
        ),
    )

    stale = (
        stale_reference_candidates(
            text_files
        )
    )

    print_section(
        "7. 구버전 API / 파일 / dependency 참조 후보",
        stale,
        empty_message=(
            "발견되지 않음"
        ),
    )

    label_duplicates = (
        duplicate_label_definitions(
            trees
        )
    )

    print_section(
        "8. labels.py 외부의 라벨 상수 재정의",
        label_duplicates,
        empty_message=(
            "발견되지 않음"
        ),
    )

    structure = (
        file_structure_issues()
    )

    print_section(
        "9. 파일 구조 문제",
        structure,
        empty_message=(
            "발견되지 않음"
        ),
    )

    security = (
        git_security_issues()
    )

    print_section(
        "10. Git / Secret 보안 점검",
        security,
        empty_message=(
            "발견되지 않음"
        ),
    )

    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    hard_failures = (
        len(parse_errors)
        + len(missing_requirements)
        + len(label_duplicates)
        + len(structure)
        + len(security)
    )

    review_candidates = (
        len(unused_imports)
        + len(unreferenced)
        + len(duplicate_defs)
        + len(duplicated_constants)
        + len(possibly_unused_requirements)
        + len(stale)
    )

    print(
        "즉시 확인이 필요한 항목: "
        f"{hard_failures}"
    )

    print(
        "수동 검토 후보: "
        f"{review_candidates}"
    )

    if hard_failures == 0:
        print(
            "구조상 치명적인 Audit 문제는 "
            "발견되지 않았습니다."
        )
    else:
        print(
            "최종 정리 전에 위의 즉시 확인 "
            "항목을 처리해야 합니다."
        )

    print()
    print(
        "주의: 미사용 import / 함수 / 상수 결과는 "
        "정적 분석 기반 후보입니다."
    )

    print(
        "출력만 보고 자동 삭제하지 마십시오."
    )


if __name__ == "__main__":
    main()