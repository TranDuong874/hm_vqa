from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="HM-VQA entry point")
    parser.add_argument(
        "--message",
        default="HM-VQA source tree was reset. Rebuild the pipeline from this clean scaffold.",
    )
    args = parser.parse_args()
    print(args.message)


if __name__ == "__main__":
    main()

