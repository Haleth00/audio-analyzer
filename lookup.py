"""Search the analysis library from the command line.

Examples:
    python lookup.py
    python lookup.py --key "C minor"
    python lookup.py --bpm-min 100 --bpm-max 140 --order tempo
"""

import argparse

import library


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--key", help='exact key, e.g. "C minor"')
    parser.add_argument("--bpm-min", type=float, help="minimum tempo")
    parser.add_argument("--bpm-max", type=float, help="maximum tempo")
    parser.add_argument("--order", choices=library.ORDER_COLUMNS,
                        default="file")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    tracks = library.search(key=args.key, bpm_min=args.bpm_min,
                            bpm_max=args.bpm_max, order=args.order,
                            limit=args.limit)
    if not tracks:
        print("no matching tracks in the library")
        return
    for t in tracks:
        print(f"{t['file'][:40]:40s} "
              f"{str(t['tempo_bpm'] or '?'):>7} BPM  "
              f"{str(t['key'] or '?'):9s} "
              f"{str(t['loudness_lufs'] or '?'):>7} LUFS")


if __name__ == "__main__":
    main()
