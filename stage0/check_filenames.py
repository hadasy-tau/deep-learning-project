import pandas as pd, numpy as np, re
P = r"C:/Users/hadas/git/deep-learning-project/stage1/outputs/segment_metrics.csv.gz"
df = pd.read_csv(P)

pat = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.wav$")
m = df.filename.str.extract(pat)
print("=== 1. filename structure ===")
print("rows                :", len(df))
print("match 4-field _.wav :", m[0].notna().sum(), f"({m[0].notna().mean():.6f})")
bad = df.loc[m[0].isna(), "filename"].head(10).tolist()
if bad: print("non-matching examples:", bad)

d = df.copy()
d[["f_spk","f_sess","f_start","f_end"]] = m.astype("float64")
d = d[d.f_spk.notna()].copy()
for c in ["f_spk","f_sess","f_start","f_end"]: d[c] = d[c].astype("int64")

print("\n=== 2. field 1 == speaker_id ===")
eq = (d.f_spk == d.speaker_id)
print("agree:", eq.sum(), f"({eq.mean():.6f})   disagree: {(~eq).sum()}")
if (~eq).any(): print(d.loc[~eq, ["filename","speaker_id"]].head(10).to_string())

print("\n=== 3. (end - start) vs duration_s ===")
span = (d.f_end - d.f_start).astype(float)
err = span - d.duration_s
print(f"span-duration:  mean {err.mean():+.4f}  median {err.median():+.4f}  std {err.std():.4f}")
print(f"               min {err.min():+.4f}  max {err.max():+.4f}")
for t in (0.5, 1.0, 2.0):
    print(f"  |err| <= {t:>4}s : {(err.abs()<=t).mean():.6f}")
print("worst 5 offenders:")
print(d.assign(err=err).reindex(err.abs().sort_values(ascending=False).index)
        [["filename","duration_s","err"]].head(5).to_string(index=False))

print("\n=== 4. session id integrity ===")
g = d.groupby("f_sess")
spk_per_sess = g.speaker_id.nunique()
print("distinct session ids                :", d.f_sess.nunique())
print("sessions mapping to >1 speaker      :", (spk_per_sess > 1).sum())
if (spk_per_sess>1).any(): print(spk_per_sess[spk_per_sess>1].head(10))
age_per_sess = g.age.nunique()
print("sessions with >1 distinct age value :", (age_per_sess > 1).sum(), f"of {len(age_per_sess)}")
agerange = g.age.agg(lambda s: s.max()-s.min())
print(f"within-session age spread (years)   : max {agerange.max():.6f}  mean {agerange.mean():.8f}")

print("\n=== 5. do (speaker, session) segments tile a timeline without overlap? ===")
ov = 0; tot = 0; spans = []
for (s, ss), grp in d.groupby(["speaker_id","f_sess"]):
    grp = grp.sort_values("f_start")
    st, en = grp.f_start.values, grp.f_end.values
    tot += len(grp)-1
    ov += int((st[1:] < en[:-1]).sum())
    spans.append(en.max()-st.min())
print("adjacent-pair overlaps:", ov, "of", tot, "pairs")
print(f"session wall-clock span (s): median {np.median(spans):.0f}  max {np.max(spans):.0f}  "
      f"= {np.median(spans)/3600:.2f} h median")

print("\n=== 6. sessions & hours per speaker ===")
sp = d.groupby("speaker_id").agg(n_seg=("filename","size"),
                                 n_sess=("f_sess","nunique"),
                                 hours=("duration_s", lambda s: s.sum()/3600),
                                 n_ages=("age","nunique"))
print(sp.describe().to_string())
print("\nspeakers meeting >=3h AND >=8 sessions :", ((sp.hours>=3)&(sp.n_sess>=8)).sum(), "of", len(sp))
print("speakers meeting >=3h                  :", (sp.hours>=3).sum())
print("speakers meeting >=8 sessions          :", (sp.n_sess>=8).sum())

print("\n=== 7. age granularity (plan check #2) ===")
r = (d.groupby("speaker_id").age.nunique() / d.groupby("speaker_id").size())
print("nunique(age)/n_segments per speaker: ", f"median {r.median():.4f}  mean {r.mean():.4f}")
r2 = d.groupby("speaker_id").apply(lambda x: x.f_sess.nunique())
r3 = d.groupby("speaker_id").age.nunique()
print("distinct ages vs distinct sessions per speaker:")
cmp = pd.DataFrame({"n_sess": r2, "n_ages": r3}); cmp["ratio"] = cmp.n_ages/cmp.n_sess
print(cmp.describe().to_string())
step = d.groupby("speaker_id").age.apply(
    lambda s: pd.Series(np.unique(np.round(s.values,4))).diff().pipe(lambda x: x[x>1e-6].min()))
step = step.dropna()
print(f"\nmin positive age step per speaker (years): median {step.median():.5f} "
      f"= {step.median()*365.25:.1f} days;  min {step.min():.5f} = {step.min()*365.25:.1f} days")
print(f"  fraction of speakers with step < 0.5 y: {(step<0.5).mean():.4f}")

d.to_parquet(r"C:/Users/hadas/AppData/Local/Temp/claude/C--Users-hadas-git-nlp-project/885490f6-7fa4-481f-a06f-1de5dcaf3097/scratchpad/stage0/segments_with_session.parquet")
print("\nwrote segments_with_session.parquet")
