"""Build deterministic local deposit candidates for the JMIR AI v3.1 release.

The script does not publish, tag, or modify canonical inputs. It creates one
code/audit ZIP and one benchmark ZIP under ``dist/`` and writes archive hashes.
"""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FIXED_ZIP_TIME = (2026, 10, 8, 0, 0, 0)

V31_PROMPT_SOURCE = "ec0791be6305f1062aabf0810a55caed4e08db5e"
V31_AGENT_NORMALIZED_SHA256 = (
    "de3cb92eef839cfdcc2c6453938f7cbd2600ae1fe146e888c5d10e418e019dbd"
)

PROTECTED_FILES = (
    "data/benchmark_500.csv",
    "outputs/ablation_500/C4_benchmark500_v31_scored.csv",
    "outputs/ablation_500/C4_benchmark500_v31_phase2.jsonl",
    "outputs/ablation_500/ablation_report_benchmark500_FINAL.md",
)

CODE_EXPLICIT = (
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "RELEASE_MANIFEST.md",
    "configs/ablation_C1_exact.yaml",
    "configs/ablation_C2_semantic.yaml",
    "configs/ablation_C3_semantic_mod.yaml",
    "configs/ablation_C4_full_pipeline.yaml",
    "configs/bios_1m_gcp.yaml",
    "configs/phase2_llm_agent_gcp.yaml",
    "data/LICENSE_CC_BY_NC_4.0.md",
    "data/README_benchmark_500.md",
    "data/benchmark_500.csv",
    "data/benchmark_500_phrases.csv",
    "outputs/ablation_500/C3_benchmark500_scored.csv",
    "outputs/ablation_500/C4_benchmark500_v31_scored.csv",
    "outputs/ablation_500/C4_benchmark500_v31_phase2.jsonl",
    "outputs/ablation_500/ablation_report_benchmark500_FINAL.md",
    "tools/build_jmir_v31_release.py",
    "tools/compute_ablation_metrics.py",
    "tools/run_ablation_500.sh",
)

DATA_FILES = (
    "data/LICENSE_CC_BY_NC_4.0.md",
    "data/README_benchmark_500.md",
    "data/benchmark_500.csv",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n"))


def tracked_under(*prefixes: str) -> list[str]:
    command = ["git", "ls-files", "--", *prefixes]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    return [line.strip().replace("\\", "/") for line in output.splitlines()]


def validate_release_identity() -> None:
    agent = ROOT / "nyuolab-src-clean" / "llm_agent.py"
    actual = normalized_lf_sha256(agent)
    if actual != V31_AGENT_NORMALIZED_SHA256:
        raise RuntimeError(
            "llm_agent.py is not the evaluated v3.1 file from "
            f"{V31_PROMPT_SOURCE}: {actual}"
        )

    for relative in PROTECTED_FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Missing protected release input: {relative}")


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = 0o755 if executable else 0o644
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def write_archive(output: Path, top_level: str, files: list[str]) -> None:
    hashes: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(set(files)):
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(f"Missing package file: {relative}")
            data = source.read_bytes()
            hashes.append(f"{sha256(data)}  {relative}")
            member = f"{top_level}/{relative}"
            archive.writestr(
                zip_info(member, executable=relative.endswith(".sh")),
                data,
            )

        sums = ("\n".join(hashes) + "\n").encode("utf-8")
        archive.writestr(
            zip_info(f"{top_level}/SHA256SUMS.txt"),
            sums,
        )


def verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        sum_name = next(
            name for name in archive.namelist() if name.endswith("/SHA256SUMS.txt")
        )
        root = sum_name.removesuffix("SHA256SUMS.txt")
        sums = archive.read(sum_name).decode("utf-8").splitlines()
        for line in sums:
            expected, relative = line.split("  ", maxsplit=1)
            actual = sha256(archive.read(f"{root}{relative}"))
            if actual != expected:
                raise RuntimeError(f"Archive hash mismatch: {relative}")


def main() -> None:
    validate_release_identity()
    DIST.mkdir(exist_ok=True)

    code_files = list(CODE_EXPLICIT)
    code_files.extend(tracked_under("nyuolab-src-clean"))

    code_zip = DIST / "llm-bios-jmir-ai-v1.0.0-v31-code-pre-release.zip"
    data_zip = DIST / "benchmark-500-v1.0.0-pre-release.zip"

    write_archive(code_zip, "llm-bios-jmir-ai-v1.0.0-v31", code_files)
    write_archive(data_zip, "benchmark-500-v1.0.0", list(DATA_FILES))
    verify_archive(code_zip)
    verify_archive(data_zip)

    archive_hashes = (
        f"{sha256(code_zip.read_bytes())}  {code_zip.name}\n"
        f"{sha256(data_zip.read_bytes())}  {data_zip.name}\n"
    )
    (DIST / "SHA256SUMS.txt").write_text(archive_hashes, encoding="utf-8")

    print(f"built {code_zip.relative_to(ROOT)} ({code_zip.stat().st_size} bytes)")
    print(f"built {data_zip.relative_to(ROOT)} ({data_zip.stat().st_size} bytes)")
    print("verified embedded and archive-level SHA-256 manifests")
    print("pre-release only: dataset license, DOI, and final SHA remain pending")


if __name__ == "__main__":
    main()
