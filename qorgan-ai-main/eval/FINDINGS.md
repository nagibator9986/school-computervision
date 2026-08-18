# The first measurement of the bullying detector

Run 2026-07-14. **657 clips, ~97 minutes, the school's own hall cameras.** This is the first
time the detector has been run over real footage, and the first number anybody has had.

```
657 clips scanned
145 candidates raised, on 140 distinct clips   (21.3% of the corpus produced anything)

confidence distribution
    below 0.72                      22
    exactly 0.72   (skeleton REFUSED to confirm)    72
    0.85 – 0.95                     23
    0.95 – 1.00                     28

ALERTS at the shipped notify_threshold (0.85):   51
```

---

## What this corpus is, and what it is not

**Every one of these 657 clips exists because the OLD detector fired on it.** They are its
trigger clips — footage adversarially pre-selected to look like bullying, and mostly false
positives. It is the hardest possible negative set.

**So "51 alerts in 97 minutes = 31.5/hour" is a true number that implies a false
conclusion**, and it must not be quoted as a false-alarm rate. An ordinary hour of corridor
is nothing like this hour. Reporting it would be the same error as
[the performance claim that counted only what it removed](../docs/superpowers/progress-ledger.md).

**What the data does support:**

- The new detector raises **no candidate at all** on **79%** of the old detector's triggers.
- At the shipped threshold it alerts on **51 of 657** — it suppresses **92%** of what the old
  system fired on.
- **How many of those 51 are real fights is unknown.** That needs a human. It is the next step.

---

## The mandatory-skeleton rule is doing real work

72 of the 145 candidates — **half of everything the fast tier proposed** — sit at exactly
0.72. That is `cap_without_skeleton`, and it fires when the pose model **looked and refused
to confirm**.

This was checked rather than assumed. Instrumenting the pose model on the capped clips:

```
skeleton outcome on 8 capped clips:   RAN  37     (13–24 crops each, min_frames=4)
```

It is **not abstaining. It is disagreeing.** The design's central safety property — *an alert
requires two independent tiers to agree* — is measurably suppressing half the fast tier's
output, and the cap (0.72) sits below the notify threshold (0.85) by construction, so a
skeleton-unconfirmed candidate **cannot** wake a teacher.

---

## What is still unknown, and why

| question | answer |
|---|---|
| How many of the 51 alerts are real fights? | **Unknown.** Needs `qorgan eval label`. |
| How many real fights did we MISS? | **Unknown, and unmeasurable from this corpus.** A clip only exists if the OLD detector fired, so there is no footage of a fight it missed. |
| Are any of the 72 skeleton-suppressed candidates real? | **Unknown.** This is the recall question, and it is the one that matters most: a fight suppressed by the skeleton is a child nobody helped. |
| What is the real false-alarm rate, per hour of ordinary corridor? | **Unknown.** This corpus cannot answer it. It needs unbiased footage. |

The corpus has **one** confirmed fight (`1.2 - ученики … толкаться`), and its start/end times
still need a human — nothing here guesses a timestamp. **One fight is not a recall number.**

See `docs/questions-for-school.md`.

---

## Next

1. `qorgan eval label` the 51 alerts and a random sample of the 72 suppressed ones. The
   second half is not optional: it is the only way to learn whether the skeleton is
   suppressing fights along with the false positives.
2. `qorgan eval run` → PR curve → choose an operating point → `save-baseline`.
3. Only then can anyone say how well this detector works.

---

# CORRECTION, 2026-07-17: "it is disagreeing" was wrong

The numbers above stand — 72 of 145 candidates really did sit at exactly 0.72, and the pose
model really did run on them. **The gloss does not.** "It is not abstaining. It is
disagreeing" credits the skeleton with a judgement, and a second measurement on different
footage shows it has none.

The 2026-07-14 corpus was the detector's **own burst captures** — clips it had already
decided were suspicious. That cannot tell you what the skeleton does with ordinary
footage, because ordinary footage never entered it. On 2026-07-16 the school supplied 33
clips they had described by hand. Measured on those, through the same production path:

| clip | the human's description | skeleton confirms |
|---|---|---|
| clip20 | "обычный проход детей по двое" | **96.7%** |
| clip09 | "подозрение на буллинг, хватает за голову" | **26.7%** |

**Ordinary walking confirms 3.6× more often than the one real incident.** The confirm/refuse
decision is driven by how many people stand in the crop and by keypoint noise, not by what
anybody did:

- Confirmation reduces, in practice, to one bit: **did any hip-centre jump 28 crop px**.
  The three WEAK reasons cannot confirm alone (`only_weak_evidence` blocks them), and
  `sudden_body_displacement`'s own 0.15 cannot reach the 0.45 bar — so the always-on weak
  noise supplies the missing weight and the displacement flips the bit.
- Pure keypoint noise, measured via torso length (a bone: it cannot change between frames,
  so its frame-to-frame delta *is* the error), has median 3.3 px. But the feature asks
  "did **any** of ~70 samples exceed 28", so noise alone predicts **82%** firing on clip20
  and **3%** on clip09. Measured: 96.7% and 26.7%. **The noise model explains nearly all of
  it.** The difference between the two clips is 5 people in the crop versus 2.
- The head-grab's pose data is **better** on every axis (keypoint confidence 0.87 vs 0.80,
  noise 1.94 px vs 3.27). The skeleton is not measuring image quality. It measures speed
  and crowding — and walking is faster than bullying, a corridor more crowded than an
  assault.

So the veto is real and what it vetoes is arbitrary. **Half the fast tier being suppressed
was never evidence that the right half was suppressed**, and this file said it was.

Also found, and worse: `body_fall_or_low_posture` is the hierarchy's only CLEAN reason —
"sufficient by itself" per `REWRITE_SPEC` §5.1 and `validation.py:37`. It weighs 0.20
against a 0.45 bar. **A child falling over, and nothing else, has never once been able to
confirm.** True in the documentation layer, silently false in the arithmetic.

Scope: 4 of the 33 clips instrumented; one school, one day; clips are 20 fps against
production's 10, so every delta measured here is roughly half production's. That biases the
firing rates and does not touch the mechanism.

Do not tune a skeleton threshold on this evidence. The numbers are denominated in crop
pixels, and the crop scale swings 0.55×–7.8× between pairs — the same threshold means
different physical motion on every pair. See `docs/client-note-2026-07-17.md`.
