"""stage0_gate - Step 0 check #6 of the Stage-2 plan.

Question: given N minutes of personal-test for a candidate speaker, what is the
smallest base-vs-tuned WER reduction we could reliably detect?

Design: both systems scored on the SAME segments; statistic is the paired
dWER = WER_base - WER_tuned; paired bootstrap over segments (2000 reps).

Simulation: the tuned model removes a fraction p of base's word errors, with
removals CLUSTERED within segments.  Clustering strength phi is ESTIMATED per
speaker from that speaker's real A/B pair, not assumed.  Draws are
beta-binomial so delta_i lies in [0, errors_i] and is unbiased.
phi = 1 is independent-word removal (optimistic floor); phi -> mean errors per
segment is fully segment-level improvement (pessimistic ceiling).
"""
import numpy as np, pandas as pd
RNG=np.random.default_rng(0); N_BOOT=2000
BUDGETS=[15,30,45,60,90,120]; ZCRIT=1.959964+0.841621   # alpha=.05 two-sided, 80% power

d=pd.read_parquet("segments_with_split.parquet")
d["err_A"]=d.S_A+d.D_A+d.I_A; d["err_B"]=d.S_B+d.D_B+d.I_B

def phi_hat(seg):
    """Overdispersion of per-segment error change, from the real A/B pair."""
    eA,eB=seg.err_A.values.astype(float),seg.err_B.values.astype(float)
    p=1-eB.sum()/eA.sum(); v=(eA*p*(1-p)).sum()
    return np.nan if v<=0 else ((eA-eB-p*eA)**2).sum()/v

def se_delta(seg,budget_min,p,phi,n_boot=N_BOOT):
    e,w,dur=(seg.err_B.values.astype(float),seg.n_ref_words.values.astype(float),seg.duration_s.values)
    k=max(2,int(round(budget_min*60/dur.mean())))
    idx=RNG.integers(0,len(seg),size=(n_boot,k)); E=e[idx]; W=w[idx].sum(1)
    rho=np.clip((phi-1)/max(e.mean()-1,1e-9),1e-9,0.9999)   # beta-binomial intra-segment corr
    s=1/rho-1
    pi=RNG.beta(max(p*s,1e-9),max((1-p)*s,1e-9),size=E.shape)
    dl=RNG.binomial(E.astype(int),pi).sum(1)/W
    return dl.mean(),dl.std(ddof=1),k,W.mean()

def mde(seg,budget_min,phi):
    lo,hi=1e-4,0.95
    for _ in range(30):
        p=(lo+hi)/2; m,s,k,wd=se_delta(seg,budget_min,p,phi)
        if m>=ZCRIT*s: hi=p
        else: lo=p
    wer=seg.err_B.sum()/seg.n_ref_words.sum()
    return dict(budget_min=budget_min,k_seg=k,words=wd,wer_B=wer,mde_rel=hi,mde_abs=hi*wer)

def se_unpaired(seg,b,n=N_BOOT):
    e,w,dur=(seg.err_B.values.astype(float),seg.n_ref_words.values.astype(float),seg.duration_s.values)
    k=max(2,int(round(b*60/dur.mean()))); i=RNG.integers(0,len(seg),size=(n,k))
    return (e[i].sum(1)/w[i].sum(1)).std(ddof=1)

PANEL={"S1 high wer_B, low gain":11835,"S2 high wer_B, normal gain":528,
       "S3 median wer_B":1057,"S4 low wer_B":4416}
print("=== per-speaker clustering, estimated from that speaker's real A/B pair ===")
PHI={}
for lab,spk in PANEL.items():
    seg=d[d.speaker_id==spk]; ph=phi_hat(seg); cap=seg.err_B.mean()
    PHI[spk]=min(ph,cap)
    print(f"  {lab:28s} spk {spk:>6}  phi_hat {ph:6.1f}  cap(mean err/seg) {cap:6.1f}"
          f"  -> phi used {PHI[spk]:6.1f}{'  [capped]' if ph>cap else ''}")

rows=[]
for lab,spk in PANEL.items():
    seg=d[d.speaker_id==spk]
    for b in BUDGETS:
        r=mde(seg,b,PHI[spk]); su=se_unpaired(seg,b)
        r.update(panel=lab,speaker_id=spk,ci95_hw_wer=1.959964*su,
                 mde_rel_unpaired=ZCRIT*np.sqrt(2)*su/r["wer_B"])
        rows.append(r)
g=pd.DataFrame(rows); pd.set_option("display.width",200)
print("\n=== MINIMUM DETECTABLE RELATIVE WER REDUCTION (paired, alpha=.05, 80% power) ===\n")
for lab in g.panel.unique():
    s=g[g.panel==lab]
    print(f"{lab}  |  speaker {s.speaker_id.iloc[0]}  |  WER_B {s.wer_B.iloc[0]:.4f}")
    print(s[["budget_min","k_seg","words","ci95_hw_wer","mde_abs","mde_rel","mde_rel_unpaired"]]
           .to_string(index=False,float_format=lambda x:f"{x:.4f}")); print()
print("=== median across panel ===")
print(g.pivot_table(index="budget_min",values=["mde_abs","mde_rel","mde_rel_unpaired"],aggfunc="median")
       .to_string(float_format=lambda x:f"{x:.4f}"))

print("\n=== sensitivity of mde_rel at 45 min to the clustering assumption ===")
print("   (phi=1 independent-word floor; 'fitted' = per-speaker estimate above; capped at mean err/seg)")
sens=[]
for phi in [1,3,6,12,20]:
    row={"phi":phi}
    for lab,spk in PANEL.items():
        seg=d[d.speaker_id==spk]; cap=seg.err_B.mean()
        row[lab.split()[0]]=mde(seg,45,min(phi,cap))["mde_rel"] if phi<=cap else np.nan
    sens.append(row)
row={"phi":"fitted"}
for lab,spk in PANEL.items(): row[lab.split()[0]]=mde(d[d.speaker_id==spk],45,PHI[spk])["mde_rel"]
sens.append(row)
print(pd.DataFrame(sens).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
g.to_csv("stage0_gate_results.csv",index=False); print("\nwrote stage0_gate_results.csv")
