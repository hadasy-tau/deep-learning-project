# deep-learning-project

**Toward Personalized Hebrew ASR: Adapting a General Model to a Single Speaker**

Automatic speech recognition is usually judged by a single average word error rate, but that average hides the fact that some speakers are transcribed well and others poorly. Using the VoxKnesset dataset (2,300 hours from 393 identified Knesset speakers), we first map per-speaker error rates for two Whisper-large-v3 models and analyze what predicts them, then fine-tune the model on the individual speakers it serves worst. The goal is a practical recipe for personalizing Hebrew ASR: which fine-tuning strategy to use, how much of a single speaker's audio it takes, and whether one adaptation can be shared across similar speakers.

Team: Dolev Abudi, Hadas Yonat.
