import argparse
import re
from pathlib import Path

import pandas as pd

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "abilene"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Abilene topology into gateway and link CSV files."
    )
    parser.add_argument(
        "--topology_file",
        type=Path,
        default=RAW_DIR / "topo-2003-04-10.txt",
        help="Path to topo-2003-04-10.txt.",
    )
    parser.add_argument(
        "--links_file",
        type=Path,
        default=RAW_DIR / "links",
        help="Path to Abilene links metadata file.",
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def read_tokens(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}. "
            "Run python src/download_abilene.py first."
        )
    return path.read_text(encoding="utf-8", errors="replace").split()


def parse_gateways(tokens: list[str]) -> pd.DataFrame:
    if "router" not in tokens or "link" not in tokens:
        raise ValueError("Topology file must contain both 'router' and 'link' sections.")

    router_index = tokens.index("router")
    link_index = tokens.index("link", router_index + 1)
    router_tokens = tokens[router_index + 1 : link_index]

    rows = []
    index = 0
    while index + 3 < len(router_tokens):
        name, city, latitude, longitude = router_tokens[index : index + 4]
        if (
            not name.startswith("#")
            and not city.startswith("#")
            and is_float(latitude)
            and is_float(longitude)
        ):
            rows.append(
                {
                    "node_id": len(rows),
                    "name": name,
                    "city": city,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                }
            )
            index += 4
        else:
            index += 1

    gateways = pd.DataFrame(rows)
    if gateways.empty:
        raise ValueError("No Abilene router/gateway rows could be parsed.")
    return gateways


def parse_topology_links(tokens: list[str], gateway_names: set[str]) -> pd.DataFrame:
    if "link" not in tokens:
        raise ValueError("Topology file does not contain a 'link' section.")

    link_index = tokens.index("link")
    link_tokens = tokens[link_index + 1 :]

    rows = []
    index = 0
    while index + 3 < len(link_tokens):
        src, dst, capacity, ospf_weight = link_tokens[index : index + 4]
        if (
            src in gateway_names
            and dst in gateway_names
            and is_float(capacity)
            and is_float(ospf_weight)
        ):
            rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "capacity_kbps": int(float(capacity)),
                    "ospf_weight": int(float(ospf_weight)),
                }
            )
            index += 4
        else:
            index += 1

    links = pd.DataFrame(rows)
    if links.empty:
        raise ValueError("No Abilene internal link rows could be parsed from topology.")
    return links


def parse_links_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: links metadata file not found, continuing without it: {path}")
        return pd.DataFrame(columns=["src", "dst", "link_index", "link_type"])

    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"([A-Za-z0-9*-]+),([A-Za-z0-9*-]+)\s+(\d+)\s+(\d+)")
    rows = [
        {
            "src": match.group(1),
            "dst": match.group(2),
            "link_index": int(match.group(3)),
            "link_type": int(match.group(4)),
        }
        for match in pattern.finditer(text)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    processed_dir = ensure_dir(args.processed_dir)

    tokens = read_tokens(args.topology_file)
    gateways = parse_gateways(tokens)
    gateway_names = set(gateways["name"].tolist())
    links = parse_topology_links(tokens, gateway_names)

    links_metadata = parse_links_metadata(args.links_file)
    if not links_metadata.empty:
        links = links.merge(
            links_metadata,
            on=["src", "dst"],
            how="left",
            validate="one_to_one",
        )

    gateways_output_path = processed_dir / "abilene_gateways.csv"
    links_output_path = processed_dir / "abilene_links.csv"
    gateways.to_csv(gateways_output_path, index=False)
    links.to_csv(links_output_path, index=False)

    print("Abilene topology parsing result:")
    print(f"- Gateways parsed: {len(gateways)}")
    print(f"- Directed internal links parsed: {len(links)}")
    if len(gateways) != 12:
        raise RuntimeError(f"Expected 12 Abilene gateways, got {len(gateways)}")

    print("\nFirst 5 gateways:")
    print(gateways[["name", "city", "latitude", "longitude"]].head().to_string(index=False))

    print("\nOutput files:")
    print(f"- {gateways_output_path}")
    print(f"- {links_output_path}")
    print(
        "\nNote: Abilene nodes are interpreted as 12 ground PoP/gateways "
        "in this project, not ordinary user terminals."
    )


if __name__ == "__main__":
    main()
