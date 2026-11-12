"""Harmonic-mix playlist builder over the analysis library.

DJ rules, encoded: consecutive tracks should be beatmatchable (tempo
within ~6%) and harmonically compatible on the Camelot wheel -- same
key, its relative major/minor, or one step either way.
"""

import argparse

import library

TEMPO_TOLERANCE = 0.06  # 6% pitch range DJs consider fair game

# Circle-of-fifths position of each major key. The relative minor of
# each sits at the same position: C major and A minor are both 8.
CAMELOT_MAJOR = {
    "C": 8, "G": 9, "D": 10, "A": 11, "E": 12,
    "B": 1, "F#": 2, "C#": 3, "G#": 4, "D#": 5, "A#": 6, "F": 7,
}

PITCH_ORDER = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

KEY_COST_WORDS = {0: "same key", 1: "relative", 2: "adjacent"}


def camelot(key):
    """'C major' -> '8B', 'A minor' -> '8A' (the Camelot wheel).

    A minor key's wheel position is its relative major's position:
    three semitones up (A minor borrows C major's 8).
    """
    pitch, mode = key.split()
    if mode == "minor":
        pitch = PITCH_ORDER[(PITCH_ORDER.index(pitch) + 3) % 12]
    return f"{CAMELOT_MAJOR[pitch]}{'A' if mode == 'minor' else 'B'}"


def _key_cost(a, b):
    """Compatibility cost between two Camelot codes.

    0 = same key, 1 = relative major/minor, 2 = adjacent on the
    wheel, None = not harmonically compatible.
    """
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    if na == nb and la == lb:
        return 0
    if na == nb:
        return 1
    if (na - nb) % 12 in (1, 11):
        return 2
    return None


def build_playlist(start_file, length=8, db_path=library.DB_PATH):
    """Greedily grow a mix starting at `start_file`.

    Each next track is the compatible candidate with the smallest
    tempo gap (ties broken by key closeness); stops early when
    nothing compatible is left. Returns {"tracks": [...],
    "transitions": [...]} with the why of every step.
    """
    tracks = [t for t in library.search(limit=10000, db_path=db_path)
              if t["tempo_bpm"] and t["key"]]
    start = next((t for t in tracks if t["file"] == start_file), None)
    if start is None:
        raise ValueError(f"{start_file!r} is not in the library "
                         f"(analyze it first)")

    for t in tracks:
        t["camelot"] = camelot(t["key"])

    playlist, used = [start], {start_file}
    while len(playlist) < length:
        current = playlist[-1]
        best = None
        for t in tracks:
            if t["file"] in used:
                continue
            tempo_gap = (abs(t["tempo_bpm"] - current["tempo_bpm"])
                         / current["tempo_bpm"])
            key_cost = _key_cost(current["camelot"], t["camelot"])
            if key_cost is None or tempo_gap > TEMPO_TOLERANCE:
                continue
            score = (tempo_gap, key_cost)  # tuples compare in order
            if best is None or score < best[0]:
                best = (score, t)
        if best is None:
            break  # nothing compatible left
        playlist.append(best[1])
        used.add(best[1]["file"])

    transitions = []
    for a, b in zip(playlist, playlist[1:]):
        gap = (b["tempo_bpm"] - a["tempo_bpm"]) / a["tempo_bpm"] * 100
        cost = _key_cost(a["camelot"], b["camelot"])
        transitions.append({
            "from": a["file"],
            "to": b["file"],
            "tempo_change_pct": round(gap, 1),
            "key_move": f"{a['camelot']} -> {b['camelot']} "
                        f"({KEY_COST_WORDS[cost]})",
        })
    return {"tracks": playlist, "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(
        description="Build a harmonic DJ mix from the library.",
    )
    parser.add_argument("start",
                        help="library filename to start the mix from")
    parser.add_argument("--length", type=int, default=8,
                        help="target number of tracks (default 8)")
    args = parser.parse_args()

    try:
        result = build_playlist(args.start, length=args.length)
    except ValueError as error:
        raise SystemExit(error)

    first = result["tracks"][0]
    print(f"Mix starting from {first['file']!r} "
          f"({first['camelot']}, {first['tempo_bpm']} BPM)\n")
    for i, t in enumerate(result["tracks"], start=1):
        if i > 1:
            step = result["transitions"][i - 2]
            print(f"     ↳ {step['tempo_change_pct']:+.1f}% tempo · "
                  f"{step['key_move']}")
        print(f"  {i}. {t['file']}  "
              f"({t['tempo_bpm']} BPM, {t['camelot']})")


if __name__ == "__main__":
    main()
