"""Stage 4 providers: one call shape over two inference back ends.

    transcribe(audio_bytes) -> Result(text, latency_s, raw)

Arm A  openai/whisper-large-v3          HF Inference Providers, provider=deepinfra
Arm B  ivrit-ai/whisper-large-v3-turbo-ct2   RunPod serverless, ivrit.ai's own image

Both were probed with the same committee chunk before this file was written
(2026-09-11); the contracts below are what the probes returned, not what the docs
say.

RunPod.  The worker is ivrit-ai/runpod-serverless infer.py.  It wants
`input.transcribe_args.blob` (base64), not `input.data` -- the latter fails with
"transcribe_args field not provided".  Non-streaming, `output[0]['result']` is a
list of batches, each a list of {type, data}; `type` is 'progress' or 'segments',
and only the latter carries text.  Arm B's turbo-ct2 is what ivrit.ai recommend
for inference; the full ivrit-ai/whisper-large-v3 is the training checkpoint.

HF.  `InferenceClient(provider='deepinfra').automatic_speech_recognition` returns
text only for this model (chunks=None); same provider and price Stage 1 used, so
Arm A stays comparable across the two corpora.

Secrets come from the environment first, then the repo-level cache/ (mode 600,
git-ignored).  Nothing here writes a key anywhere.
"""
import base64, os, time
from dataclasses import dataclass, field

import requests

HERE  = os.path.dirname(os.path.abspath(__file__))
# Repo-level cache/, not any one builder's.  These secrets used to live in the
# speaker-index builder's cache/, which they reached for only because it was
# already git-ignored.  They belong to the repo.
CACHE = os.path.join(HERE, '..', '..', 'cache')

def secret(env, cache_file):
    v = os.environ.get(env)
    p = os.path.join(CACHE, cache_file)
    if not v and os.path.exists(p):
        v = open(p).read().strip()
    if not v:
        raise RuntimeError(f'missing secret: set {env} or create {p}')
    return v

@dataclass
class Result:
    text: str
    latency_s: float
    raw: dict = field(default_factory=dict, repr=False)

class TransientError(Exception):
    """Retry-worthy: rate limit, queue timeout, 5xx, network."""

class Provider:
    name = arm = model = None
    def transcribe(self, audio: bytes) -> Result:
        raise NotImplementedError

# ---------------------------------------------------------------------------
class HFProvider(Provider):
    name, arm, model = 'deepinfra', 'A', 'openai/whisper-large-v3'
    def __init__(self, provider='deepinfra', model=None, token=None):
        from huggingface_hub import InferenceClient
        self.name = provider
        self.model = model or self.model
        self.client = InferenceClient(provider=provider, api_key=token or secret('HF_INFERENCE_TOKEN', 'hf_inference_token'))
    def transcribe(self, audio):
        t0 = time.perf_counter()
        try:
            out = self.client.automatic_speech_recognition(audio=audio, model=self.model)
        except Exception as e:
            s = str(e)
            if any(k in s for k in ('429', '503', '502', '504', 'Timeout', 'timed out', 'Connection')):
                raise TransientError(s) from e
            raise
        return Result(text=(out.text or '').strip(), latency_s=time.perf_counter() - t0,
                      raw={'chunks': None if out.chunks is None else [c.__dict__ for c in out.chunks]})

# ---------------------------------------------------------------------------
class RunPodProvider(Provider):
    name, arm, model = 'runpod', 'B', 'ivrit-ai/whisper-large-v3-turbo-ct2'
    MAX_PAYLOAD = 10 * 1024 * 1024          # the ivrit client's own ceiling
    def __init__(self, model=None, engine='faster-whisper', language='he', api_key=None, endpoint_id=None, timeout=300):
        self.model = model or self.model
        self.engine, self.language, self.timeout = engine, language, timeout
        key = api_key or secret('RUNPOD_API_KEY', 'runpod_api_key')
        self.endpoint = endpoint_id or secret('RUNPOD_ENDPOINT_ID', 'runpod_endpoint_id')
        self.base = f'https://api.runpod.ai/v2/{self.endpoint}'
        self.headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    def health(self):
        r = requests.get(f'{self.base}/health', headers=self.headers, timeout=30); r.raise_for_status()
        return r.json()

    def payload(self, audio):
        return {'input': {'type': 'blob', 'model': self.model, 'engine': self.engine, 'streaming': False,
                          'transcribe_args': {'language': self.language, 'diarize': False, 'diarization_args': None,
                                              'output_options': {}, 'verbose': False,
                                              'blob': base64.b64encode(audio).decode()}}}

    @staticmethod
    def parse(output):
        """Concatenate the text of every 'segments' entry, in order; skip 'progress'."""
        if not output or not isinstance(output, list):
            raise ValueError(f'unexpected output: {str(output)[:200]}')
        segs = [s for batch in output[0].get('result', []) for e in batch
                if e.get('type') == 'segments' for s in e.get('data', [])]
        return ''.join(s.get('text', '') for s in segs).strip(), segs

    def transcribe(self, audio):
        """Submit to /run and poll /status.  A blocking /runsync ties one caller
        to one job for its whole queue wait; on the 10-chunk probe the GPU did
        0.8 s of work inside a 7.7 s median call (17%), the rest queue and
        transfer.  Async submit lets the caller keep the endpoint's queue full
        with more in-flight jobs than workers, so that wait is overlapped."""
        p = self.payload(audio)
        if len(p['input']['transcribe_args']['blob']) > self.MAX_PAYLOAD:
            raise ValueError(f'audio too large for RunPod blob: {len(audio):,} B')
        t0 = time.perf_counter()
        try:
            r = requests.post(f'{self.base}/run', headers=self.headers, json=p, timeout=60)
        except requests.RequestException as e:
            raise TransientError(str(e)) from e
        if r.status_code in (429, 500, 502, 503, 504):
            raise TransientError(f'HTTP {r.status_code}: {r.text[:200]}')
        r.raise_for_status()
        job_id = r.json().get('id')
        if not job_id:
            raise RuntimeError(f'no job id in submit response: {r.text[:200]}')
        d = self._poll(job_id)
        status = d.get('status')
        if status != 'COMPLETED':
            err = str(d.get('error'))
            if status in ('TIMED_OUT', 'CANCELLED') or 'worker' in err.lower():
                raise TransientError(f'{status}: {err[:200]}')
            raise RuntimeError(f'RunPod {status}: {err[:300]}')
        text, segs = self.parse(d.get('output'))
        return Result(text=text, latency_s=time.perf_counter() - t0,
                      raw={'job_id': job_id, 'delay_ms': d.get('delayTime'), 'exec_ms': d.get('executionTime'),
                           'worker': d.get('workerId'), 'segments': segs})

    def _poll(self, job_id, first=0.5, every=1.0):
        """Poll /status until terminal.  Short first wait (most jobs finish in
        ~1 s of GPU time), then 1 s cadence; errors on the poll itself retry."""
        deadline = time.time() + self.timeout; wait = first; d = {}
        while time.time() < deadline:
            time.sleep(wait); wait = every
            try:
                r = requests.get(f'{self.base}/status/{job_id}', headers=self.headers, timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 429:
                time.sleep(3); continue
            r.raise_for_status(); d = r.json()
            if d.get('status') not in ('IN_QUEUE', 'IN_PROGRESS'):
                return d
        try:
            requests.post(f'{self.base}/cancel/{job_id}', headers=self.headers, timeout=15)
        except requests.RequestException:
            pass
        raise TransientError(f'job {job_id} still {d.get("status")} after {self.timeout}s')

ARMS = {'A': HFProvider, 'B': RunPodProvider}

def make(arm, **kw):
    return ARMS[arm](**kw)

# ---- self-checks (no network) ------------------------------------------------
if __name__ == '__main__':
    fake = [{'result': [[{'type': 'progress', 'data': {}},
                         {'type': 'segments', 'data': [{'text': ' שלום', 'start': 0, 'end': 1}]}],
                        [{'type': 'segments', 'data': [{'text': ' לכולם.', 'start': 1, 'end': 2}]}]]}]
    text, segs = RunPodProvider.parse(fake)
    assert text == 'שלום לכולם.' and len(segs) == 2, (text, segs)
    try:
        RunPodProvider.parse(None); raise AssertionError('should reject None')
    except ValueError:
        pass
    p = RunPodProvider.__new__(RunPodProvider); p.model = 'm'; p.engine = 'faster-whisper'; p.language = 'he'
    pl = p.payload(b'abc')
    assert pl['input']['transcribe_args']['blob'] == base64.b64encode(b'abc').decode()
    assert pl['input']['streaming'] is False and 'data' not in pl['input']
    print('providers.py self-checks OK')
