import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w", encoding="utf-8") as fout:
        for inp in args.inputs:
            p = Path(inp)
            if not p.exists():
                raise FileNotFoundError(p)
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n += 1

    print(f"[Merged] {n} lines -> {out}")


if __name__ == "__main__":
    main()
