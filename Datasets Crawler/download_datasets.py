from __future__ import annotations

import csv
import hashlib
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datasets.csv"
OUTPUT_DIR = BASE_DIR / "datasets"
REPORT_PATH = BASE_DIR / "下载报告.md"
TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
CHUNK_SIZE = 64 * 1024
REQUIRED_COLUMNS = {"id", "name", "source", "url", "filename"}


def create_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    # Python 3.13+ enables strict X.509 checks that reject LIBSVM's otherwise
    # trusted certificate because an intermediate lacks a legacy extension.
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


SSL_CONTEXT = create_ssl_context()


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def load_datasets() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV 缺少字段: {', '.join(sorted(missing))}")

        datasets = []
        seen_ids = set()
        seen_filenames = set()
        for line_number, row in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in row.items()}
            if not any(row.values()):
                continue
            if not all(row[column] for column in REQUIRED_COLUMNS):
                raise ValueError(f"CSV 第 {line_number} 行存在必填字段空值")
            if row["id"] in seen_ids:
                raise ValueError(f"CSV 第 {line_number} 行 ID 重复: {row['id']}")
            if row["filename"] in seen_filenames:
                raise ValueError(f"CSV 第 {line_number} 行文件名重复: {row['filename']}")
            if Path(row["filename"]).name != row["filename"]:
                raise ValueError(f"CSV 第 {line_number} 行文件名不安全: {row['filename']}")
            seen_ids.add(row["id"])
            seen_filenames.add(row["filename"])
            datasets.append(row)

    if not datasets:
        raise ValueError("CSV 中没有数据集")
    return datasets


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def download(dataset: dict[str, str]) -> dict[str, object]:
    destination = OUTPUT_DIR / dataset["filename"]
    temporary = destination.with_name(destination.name + ".part")
    last_error = "未知错误"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(
                dataset["url"],
                headers={"User-Agent": "OVO-OVR-Dataset-Crawler/1.0"},
            )
            with urllib.request.urlopen(
                request, timeout=TIMEOUT_SECONDS, context=SSL_CONTEXT
            ) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP {status}")
                with temporary.open("wb") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)

            if size == 0:
                raise RuntimeError("服务器返回了空文件")
            temporary.replace(destination)
            return {
                **dataset,
                "success": True,
                "size": size,
                "sha256": digest.hexdigest(),
                "message": "下载成功",
            }
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = str(error).replace("\n", " ").replace("|", "\\|")
            temporary.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 2)

    return {
        **dataset,
        "success": False,
        "size": 0,
        "sha256": "",
        "message": last_error,
    }


def write_report(results: list[dict[str, object]], started_at: datetime) -> None:
    finished_at = datetime.now().astimezone()
    succeeded = sum(bool(result["success"]) for result in results)
    failed = len(results) - succeeded
    lines = [
        "# 数据集下载报告",
        "",
        f"- 开始时间：{started_at.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 完成时间：{finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- CSV：`{CSV_PATH.name}`",
        f"- 下载目录：`datasets/`",
        f"- 总计：{len(results)}",
        f"- 成功：{succeeded}",
        f"- 失败：{failed}",
        "",
        "## 下载结果",
        "",
        "| 状态 | ID | 数据集 | 来源 | 文件 | 大小 | SHA-256 | 说明 |",
        "|---|---|---|---|---|---:|---|---|",
    ]

    for result in results:
        status = "成功" if result["success"] else "失败"
        size = format_size(int(result["size"])) if result["success"] else "-"
        checksum = f"`{result['sha256']}`" if result["success"] else "-"
        filename = f"[`{result['filename']}`](datasets/{result['filename']})" if result["success"] else f"`{result['filename']}`"
        lines.append(
            f"| {status} | {result['id']} | {result['name']} | {result['source']} | "
            f"{filename} | {size} | {checksum} | {result['message']} |"
        )

    failed_results = [result for result in results if not result["success"]]
    lines.extend(["", "## 失败数据集", ""])
    if failed_results:
        for result in failed_results:
            lines.append(f"- `{result['id']}` {result['name']}：{result['message']}")
    else:
        lines.append("全部数据集下载成功。")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 每次运行会先删除整个 `datasets/` 文件夹，再重新下载 CSV 中的全部数据集。",
            "- 报告中的论文规模来自 CSV，仅用于版本识别；下载器不解析或清洗数据内容。",
            "- SHA-256 可用于确认后续使用的数据文件是否与本次下载一致。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    started_at = datetime.now().astimezone()
    try:
        datasets = load_datasets()
        reset_output_dir()
    except (OSError, ValueError) as error:
        print(f"初始化失败: {error}", file=sys.stderr)
        return 2

    print(f"将下载 {len(datasets)} 个数据集到: {OUTPUT_DIR}")
    results = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index:02d}/{len(datasets):02d}] {dataset['id']} {dataset['name']} ... ", end="", flush=True)
        result = download(dataset)
        results.append(result)
        if result["success"]:
            print(f"成功 ({format_size(int(result['size']))})")
        else:
            print(f"失败 ({result['message']})")

    try:
        write_report(results, started_at)
    except OSError as error:
        print(f"报告生成失败: {error}", file=sys.stderr)
        return 2

    failed = sum(not bool(result["success"]) for result in results)
    print(f"报告已生成: {REPORT_PATH}")
    print(f"下载完成: 成功 {len(results) - failed}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
