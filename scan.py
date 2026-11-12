"""Command-line interface: scan a folder of audio files and report
their basic properties.

With --separate, also splits each file into vocals and instrumental
using the Demucs model, writing FLAC stems to separated/.

The analyses live in analyze.py and the separation in separation.py,
both shared with the web service (server.py).
"""

import argparse
import csv
from pathlib import Path

from analyze import analyze_file, find_audio_files
from library import record
from separation import separate_file


def print_result(result):
    """Print one analysis result dict in the scanner's format."""
    print(result["file"])
    print(f"  duration:    {result['duration_s']} s")
    print(f"  sample rate: {result['sample_rate_hz']} Hz")
    print(f"  channels:    {result['channels']}")
    if result["tempo_bpm"] is not None:
        print(f"  tempo:       {result['tempo_bpm']} BPM")
    else:
        print("  tempo:       could not estimate")
    if result["loudness_lufs"] is not None:
        print(f"  loudness:    {result['loudness_lufs']} LUFS")
    else:
        print("  loudness:    could not estimate")
    if result["key"] is not None:
        print(f"  key:         {result['key']} "
              f"(correlation {result['key_confidence']})")
    else:
        print("  key:         could not estimate")
    if result["chord_progression"] is not None:
        print(f"  chords:      {result['chord_progression']}")
    else:
        print("  chords:      could not estimate")
    if result["sections"]:
        print("  structure:")
        for s in result["sections"]:
            print(f"    {s['start']:6.1f}s  {s['label']} "
                  f"({s['duration_s']}s, "
                  f"brightness {s['brightness_hz']:.0f} Hz)")


CSV_FIELDS = ["file", "duration_s", "sample_rate_hz", "channels",
              "tempo_bpm", "loudness_lufs", "key", "key_confidence",
              "chord_progression"]


def write_report(results, csv_path):
    """Write analysis result dicts to a CSV file.

    DictWriter pulls the values straight out of the same dicts that
    analyze_file() produces, by key. None becomes an empty cell.
    """
    # newline="" is the csv module's own rule: it manages line endings
    # itself (so the file is identical on Windows and Linux).
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {csv_path}")


def main(folder, do_separate, csv_path, save=True):
    files = find_audio_files(folder)
    if not files:
        print(f"No audio files found in {folder}")
        return

    results = []
    for path in files:
        try:
            result = analyze_file(path)
        except RuntimeError as error:
            # A file with an audio extension that soundfile can't open.
            print(f"{path.name}: could not read ({error})")
            print()
            continue
        results.append(result)
        if save:
            try:
                record(result)
            except Exception as error:
                # The analysis is still valid; failing to remember it
                # shouldn't kill the scan.
                print(f"  library:     could not record ({error})")
        print_result(result)

        if do_separate:
            out_dir = Path("separated") / path.stem
            try:
                separate_file(path, out_dir)
                print(f"  separation:  {out_dir}/vocals.flac, no_vocals.flac")
            except Exception as error:
                print(f"  separation:  failed ({error})")

        print()

    if csv_path:
        write_report(results, csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a folder of audio files and report their properties."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="audio",
        type=Path,
        help="folder to scan (default: audio)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="also split each file into vocals and instrumental (slow)",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="report.csv",
        default=None,
        metavar="FILE",
        help="also write a CSV report of every analysis "
             "(default file name: report.csv)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="don't record analyses into the library database",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.folder, args.separate, args.csv, save=not args.no_save)
