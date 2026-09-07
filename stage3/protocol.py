"""Stage 3 protocol: load one session's text-side files, the matching corpus
shard, and put both on the same character axis.

Facts that shape this file (see stage3/plan.md):
  * transcript.refined.json['text'] is raw.protocol.txt; map.json[i] gives the
    char span of segments[i] in it; speakers.segments.txt gives each raw
    speaker's char range.  All three share one axis (create_maps.py, ivrit.ai).
  * ''.join(w['word'] for w in segment['words']) == segment['text'] held for
    15,612/15,612 segments, so per-word char offsets are exact by accumulation.
  * The corpus shard's sentences are the same text (99.6% similar); 82-100% of
    them occur verbatim.  Placement is monotonic and verbatim -- an unplaced
    sentence is skipped, never fuzzed.
  * refined.json is ~3 MB per session (words + tokens).  It is fetched into
    memory and kept only when `keep=True` (pilot sessions); the small files and
    the ~100 KB shards are always cached.
"""
import bz2, json, os, random, statistics, time
import requests
from huggingface_hub import hf_hub_download, get_token

from roster import CACHE

COMMITTEES = 'ivrit-ai/knesset-committees'
CORPUS     = 'HaifaCLGroup/KnessetCorpus'
SHARD_DIR  = 'protocols_sentences/committee_protocols/data'
HF_CACHE   = os.path.join(CACHE, 'hf')
SMALL      = ['transcript.refined.map.json', 'speakers.txt', 'speakers.segments.txt', 'metadata.json']

def _dl(repo, path, keep=True, tries=6):
    """keep=True caches via hf_hub_download (pilot sessions, re-read by validate);
    keep=False streams and discards (the full build reads each session once, and
    caching 11k refined.json would be ~33 GB).

    hf_hub_download issues a HEAD before every GET, so the four small files per
    session cost eight hub-API calls.  At twelve workers that earned a stream of
    HTTP 429s within two batches.  `resolve/main` over plain requests is one call
    per file and is not metered the same way, so the small files go through the
    same retrying fetcher as the big one.  Backoff is capped and jittered so
    workers that are throttled together do not retry in lockstep."""
    if keep:
        return open(hf_hub_download(repo, path, repo_type='dataset', token=get_token(), cache_dir=HF_CACHE), 'rb').read()
    last = None
    for i in range(tries):
        try:
            r = requests.get(f'https://huggingface.co/datasets/{repo}/resolve/main/{path}',
                             headers={'Authorization': f'Bearer {get_token()}'}, timeout=300)
            if r.status_code == 404:
                r.raise_for_status()
            if r.ok:
                return r.content
            last = requests.HTTPError(f'{r.status_code} for {path}')
        except requests.RequestException as e:
            last = e
        if i < tries - 1:
            time.sleep(min(2 ** i, 30) * (1 + random.random()))
    raise last

def load_session(session_id, keep=False):
    """Everything text-side for one session, on one char axis."""
    sid = str(session_id)
    refined = json.loads(_dl(COMMITTEES, f'{sid}/transcript.refined.json', keep))
    small = {f: _dl(COMMITTEES, f'{sid}/{f}', keep) for f in SMALL}
    speakers = {}
    for line in small['speakers.txt'].decode().splitlines():
        if '\t' in line:
            lid, name = line.split('\t', 1); speakers[int(lid)] = name
    spk_segments = [tuple(map(int, l.split('\t'))) for l in small['speakers.segments.txt'].decode().splitlines() if l.strip()]
    meta = json.loads(small['metadata.json'])
    meta.pop('per_segment_quality_scores', None)          # keyed to the aligned, not refined, segments
    return dict(session_id=int(sid), text=refined['text'], segments=refined['segments'],
                map=json.loads(small['transcript.refined.map.json']),
                speakers=speakers, spk_segments=spk_segments, meta=meta)

def load_shard(knesset, protocol_name):
    raw = _dl(CORPUS, f'{SHARD_DIR}/{knesset}/{protocol_name}.jsonl.bz2')
    return json.loads(bz2.decompress(raw).decode('utf-8'))

def place_gold(gold, text):
    """Gold sentences located verbatim in `text`, in order.  Returns
    [(start_char, end_char, speaker_id, speaker_name, is_valid)], and the
    fraction placed."""
    placed, pos, n = [], 0, 0
    for s in gold['protocol_sentences']:
        t = s['sentence_text'].strip()
        if not t:
            continue
        n += 1
        i = text.find(t, pos)
        if i < 0:
            i = text.find(t)                       # rare backward jump (repeated sentence)
        if i < 0:
            continue
        placed.append((i, i + len(t), s['speaker_id'], s['speaker_name'], bool(s.get('is_valid_speaker', True))))
        pos = i + len(t)
    return placed, (len(placed) / n if n else 0.0)

def word_table(session):
    """One record per word with its char span, time span, probability and raw
    speaker.  Returns (words, verified_fraction).

    A segment's word offsets are anchored at its own map['start_char'], so they
    are exact whenever the segment's declared position really holds its text.
    That -- not end_char arithmetic -- is the check: in a 200-session pilot three
    sessions had map spans whose lengths disagreed with the text by 1-5 chars,
    yet one of them placed every segment correctly and another placed 36 of 39.
    Words in a segment that fails the check carry offset_ok=False and are never
    given an identity; the caller drops a session whose verified fraction is low.
    """
    T = session['text']; spk = session['spk_segments']; words = []; ok_n = 0
    for i, (seg, m) in enumerate(zip(session['segments'], session['map'])):
        a0, txt = m['start_char'], seg['text']
        ok = (''.join(w['word'] for w in seg['words']) == txt) and T[a0:a0 + len(txt)] == txt
        ok_n += ok
        off = a0
        for j, w in enumerate(seg['words']):
            a, b = off, off + len(w['word']); off = b
            mid = (a + b) / 2
            lid = next((l for l, s, e in spk if s <= mid < e), None)
            words.append(dict(seg=i, idx=j, a=a, b=b, t0=w['start'], t1=w['end'],
                              p=w['probability'], word=w['word'], local_speaker_id=lid,
                              offset_ok=ok))
    return words, (ok_n / len(session['segments']) if session['segments'] else 0.0)

def gold_at(placed, a, b):
    """Gold sentence covering the word span [a, b), or None."""
    mid = (a + b) / 2
    for p in placed:
        if p[0] <= mid < p[1]:
            return p
    return None

def quality(words):
    return statistics.median(w['p'] for w in words)

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    s = load_session(2073683, keep=True)
    assert s['meta']['session_date'].startswith('2018-11-06') and s['meta']['knesset_num'] == '20'
    assert len(s['segments']) == len(s['map']) == 815
    assert s['speakers'][10000000] == 'היו"ר שלי יחימוביץ'
    words, ok = word_table(s)
    assert len(words) == 8323 and words[0]['word'] == 'בוקר' and words[0]['a'] == 0 and ok == 1.0
    assert all(w['local_speaker_id'] is not None and w['offset_ok'] for w in words)
    # a session whose map spans disagree with the text still places most segments
    bad, frac = word_table(load_session(2075781, keep=True))
    assert 0.9 < frac < 1.0, frac
    assert sum(not w['offset_ok'] for w in bad) > 0
    g = load_shard(20, '20_ptv_519810.doc')
    placed, frac = place_gold(g, s['text'])
    assert frac > 0.95, frac
    assert gold_at(placed, 0, 9)[2] == '4405'
    print(f'protocol.py self-checks OK: {len(words)} words, {len(placed)} gold sentences placed ({frac:.1%})')
