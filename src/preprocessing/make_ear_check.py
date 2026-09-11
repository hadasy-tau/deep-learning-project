"""Build the ear-check page from a completed build's part files.

The one gate `preprocessing.py` cannot pass by itself: a time-origin offset
between a session's `audio.m4a` and its alignment produces chunks that are
fluent, correctly cut, correctly labelled -- and of the wrong words.  Every
assertion in the module passes on them.  Only a person listening catches it.

This emits a single self-contained HTML page: each clip as lossless FLAC beside
the protocol text it claims to be, with a verdict control per clip.  Two clips
per session where there are two -- one near the start and one near the end,
because a constant offset shows in either but drift only shows by comparing
them.

The page it writes is ~7 MB, almost all base64 audio, and is gitignored: this
script and `ear_check_template.html` are the committed artefacts.

    python src/preprocessing/make_ear_check.py   # -> src/preprocessing/ear_check.html
"""
import base64
import json
import os

import pandas as pd

import preprocessing as P

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'ear_check_template.html')
OUT_HTML = os.path.join(HERE, 'ear_check.html')

# Clips are chosen clean on purpose.  At low quality you would be hearing an
# alignment failure the score already reports, and calling that a mismatch says
# nothing about timing -- which is the only thing this page tests.
MIN_Q, MIN_S, MAX_S = 0.5, 3.0, 10.0
FALLBACK_Q, FALLBACK_MIN_S, FALLBACK_MAX_S = 0.3, 2.0, 14.0


def load_parts(parts_dir=None):
    """Every chunk written by a build, plus its per-session stats."""
    parts_dir = parts_dir or P.PARTS
    files = [f for f in os.listdir(parts_dir) if f.endswith('.parquet')]
    if not files:
        raise SystemExit(f'no part files in {parts_dir}; run a build first')
    df = pd.concat([pd.read_parquet(os.path.join(parts_dir, f)) for f in files],
                   ignore_index=True)
    stats = {}
    for f in os.listdir(parts_dir):
        if f.endswith('_stats.json'):
            s = json.load(open(os.path.join(parts_dir, f), encoding='utf-8'))
            stats[int(s['session'])] = s
    return df, stats


def pick(df):
    """Two clips per session: the earliest and the latest that qualify."""
    out = []
    for _, g in df.groupby('session'):
        ok = g[(g.quality >= MIN_Q) & (g.duration_s.between(MIN_S, MAX_S))].sort_values('abs_start')
        if len(ok) < 2:
            ok = g[(g.quality >= FALLBACK_Q)
                   & (g.duration_s.between(FALLBACK_MIN_S, FALLBACK_MAX_S))].sort_values('abs_start')
        if len(ok) == 0:
            ok = g.sort_values('abs_start')
        out.append(ok.iloc[[0]] if len(ok) == 1 else pd.concat([ok.iloc[[0]], ok.iloc[[-1]]]))
    return pd.concat(out, ignore_index=True)


def payload(df, stats):
    clips = [{
        'id': r.chunk_id, 'session': int(r.session), 'speaker': r.speaker_name,
        'dur': round(float(r.duration_s), 2), 'q': round(float(r.quality), 3),
        'at': round(float(r.abs_start), 1), 'text': r.text,
        'audio': base64.b64encode(r.audio['bytes']).decode('ascii'),
    } for r in df.itertuples()]
    sessions = []
    for sid, g in df.groupby('session'):
        s = stats.get(int(sid), {})
        sessions.append({
            'session': int(sid), 'chunks': int(s.get('n_chunks') or len(g)),
            'decoded_s': round(float(s.get('decoded_s') or 0), 1),
            'aligned_s': round(float(s.get('max_end_s') or 0), 1),
            'dropped': int(s.get('n_dropped') or 0),
        })
    return {'clips': clips, 'sessions': sorted(sessions, key=lambda x: x['session'])}


def write(data, template=TEMPLATE, out=OUT_HTML):
    tpl = open(template, encoding='utf-8').read()
    assert '__DATA__' in tpl, f'{template} has no __DATA__ placeholder'
    # Escape '<' so nothing in the payload can close the script element early --
    # a truncated JSON block leaves the page rendering its empty shell.
    blob = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    open(out, 'w', encoding='utf-8').write(tpl.replace('__DATA__', blob))
    return out


if __name__ == '__main__':
    df, stats = load_parts()
    chosen = pick(df)
    path = write(payload(chosen, stats))
    mb = os.path.getsize(path) / 1e6
    print(f'{len(chosen)} clips from {chosen.session.nunique()} sessions '
          f'({chosen.duration_s.sum():.0f} s of audio) -> {path}  {mb:.1f} MB')
    # The artifact ceiling is 16 MB rendered; base64 is 4/3 of the bytes it carries.
    assert mb < 15, 'over the artifact size budget -- tighten MIN_S/MAX_S'
