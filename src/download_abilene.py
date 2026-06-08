import argparse
from pathlib import Path

import requests

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "abilene"
BASE_URL = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM"
STATIC_FILES = ["readme.txt", "topo-2003-04-10.txt", "links", "demands"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Abilene traffic matrices and metadata files."
    )
    parser.add_argument(
        "--start_week",
        type=int,
        default=1,
        help="First Abilene week to download, inclusive. Valid range: 1-24.",
    )
    parser.add_argument(
        "--end_week",
        type=int,
        default=24,
        help="Last Abilene week to download, inclusive. Valid range: 1-24.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=RAW_DIR,
        help="Directory for downloaded raw Abilene files.",
    )
    return parser.parse_args()


def validate_week_range(start_week: int, end_week: int) -> None:
    if not 1 <= start_week <= 24:
        raise ValueError(f"--start_week must be in [1, 24], got {start_week}")
    if not 1 <= end_week <= 24:
        raise ValueError(f"--end_week must be in [1, 24], got {end_week}")
    if start_week > end_week:
        raise ValueError(
            f"--start_week must be <= --end_week, got {start_week} > {end_week}"
        )


def build_download_list(start_week: int, end_week: int) -> list[str]:
    week_files = [f"X{week:02d}.gz" for week in range(start_week, end_week + 1)]
    return week_files + STATIC_FILES


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def download_file(filename: str, output_dir: Path) -> Path:
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"Skip existing file: {filename} ({format_size(output_path.stat().st_size)})")
        return output_path

    print(f"Downloading: {filename}")
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download {filename}: HTTP {response.status_code} from {url}"
                )

            with temp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        temp_path.replace(output_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {filename}: {exc}") from exc

    print(f"Downloaded: {filename} ({format_size(output_path.stat().st_size)})")
    return output_path


def main() -> None:
    args = parse_args()
    validate_week_range(args.start_week, args.end_week)

    raw_dir = ensure_dir(args.raw_dir)
    print(f"Raw Abilene directory: {raw_dir}")

    downloaded_paths = []
    for filename in build_download_list(args.start_week, args.end_week):
        downloaded_paths.append(download_file(filename, raw_dir))

    print("\nDownloaded / available files:")
    for path in downloaded_paths:
        print(f"- {path} ({format_size(path.stat().st_size)})")


if __name__ == "__main__":
    main()
