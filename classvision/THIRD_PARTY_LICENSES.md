# Third-party licences — read before selling this to a school

Two of the dependencies below are **delivery blockers, not footnotes**. Both are avoidable,
and both are much cheaper to deal with now than after a school has paid.

---

## 1. Ultralytics (YOLO11 / YOLO26 pose) — **AGPL-3.0**

Used by: `classvision.vision.pose`, i.e. every analysis run.

The AGPL's network clause is the issue: if a **network-served** application is a derivative
work of AGPL code, its complete corresponding source must be offered to users of that
service. A school dashboard that ran the model in-process would plausibly be exactly that.

**What this project does about it.** `ultralytics` is confined to the `analyse` extra
(`pyproject.toml`) and to a process that writes a JSON file. The web application imports
only `report/artefact.py` and never the model stack — that is the rule in `INTEGRATION.md`
§1, and §8 specifies a test that fails if `qorgan` ever imports `torch`. **This separation
is doing legal work, not just architectural tidiness, and must not be "simplified" later.**

**Options if the school wants a commercial, closed deployment:**
1. Buy an Ultralytics Enterprise licence.
2. Swap the detector for an Apache-2.0 model (RTMDet/RTMPose family). This is a real
   option — the pose interface is one class, `PoseModel` — but see §5 below: the `mmcv`
   toolchain does not install on this host, so it would need ONNX weights and a direct
   `onnxruntime` path, not the mm* stack.
3. Keep the offline-file architecture and accept AGPL for the batch tool only.

## 2. InsightFace `buffalo_l` weights — **non-commercial research use only**

Used by: `classvision.identity.faces`, i.e. the optional face-corroboration pass.

The InsightFace **code** is MIT. The **`buffalo_l` model pack is not** — it is released for
academic/non-commercial use. An MIT code licence does not launder the weights.

**What this project does about it.** Face corroboration is **off by default**
(`Settings.face_corroboration = False`), it lives in its own `faces` extra, and it cannot
create a name under any configuration — the measured evidence (median best cosine 0.30,
margin 0.10; `MEASUREMENTS.md` §4) does not support identification, so its only role is to
agree or disagree with a human's seating plan. **A school deployment can therefore ship
with this feature simply switched off and lose nothing that is currently trustworthy.**

If face corroboration is ever wanted commercially, the weights must be replaced with a
commercially-licensed recogniser and the numbers in `MEASUREMENTS.md` §4 re-measured — they
are properties of that specific model.

## 3. Everything else

| package | licence | notes |
|---|---|---|
| PyTorch | BSD-3-Clause | permissive |
| OpenCV (`opencv-python`) | Apache-2.0 | permissive. Do **not** add `opencv-contrib-python` alongside it — two `cv2` modules in one environment |
| NumPy, SciPy | BSD-3-Clause | permissive |
| scikit-learn | BSD-3-Clause | permissive. DBSCAN in `room/seats.py` |
| Matplotlib | PSF-based (BSD-compatible) | permissive |
| ONNX Runtime | MIT | permissive |

## 4. Model weight provenance

Record the SHA-256 of every weight file a deployment actually loads, next to the `imgsz`
and version already captured in `Provenance.model`. A metric compared across two terms
under two different checkpoints is a comparison of checkpoints.

```bash
shasum -a 256 yolo11m-pose.pt ~/.insightface/models/buffalo_l/*.onnx
```

## 5. Things deliberately NOT used, and the licence reason

Researched and rejected — recorded so the question is not re-opened every six months:

| candidate | why not |
|---|---|
| Sapiens (Meta) | licence terms unsuitable for this deployment |
| 6DRepNet / WHENet / HopeNet (head pose) | trained on **300W-LP: research-only**. Also unusable here on measurement grounds — a 56 px ear-span upscaled to a 224² crop adds no information |
| Gaze360 / ETH-XGaze (gaze) | non-commercial **and** physically impossible here: measured inter-pupil distance 18.8 px, so an iris is 1–2 px |
| VideoMAE / VideoMAE V2 | `cc-by-nc-4.0`, non-commercial |
| SCB-Dataset (classroom behaviour) | "academic research, personal learning and non-commercial use only". Genuinely covers 4 of our 5 states — acceptable for a diploma experiment, blocked for a product |
| mmaction2 / mmcv | not a licence issue: **`mmcv` has zero cp313/arm64 distributions**, and `mmaction2` requires `decord`, last released 2021 with x86-only macOS wheels. Recorded as a negative result — it is part of why hand-written predicates are a deliberate choice rather than a shortcut |

---

## 6. The non-licence legal constraints that bind this system

Not third-party terms, but they belong next to them because they constrain the same
delivery decision.

* **Emotion inference in education is prohibited outright** under the EU AI Act
  Art. 5(1)(f). This system has no expression classifier and no place to add one, and
  `states.py` and `metrics/activity.py` both state in their module docstrings that
  «вовлечённость» is not measured and not inferred. **Never add affect recognition.** If
  the school is not in EU scope this is still the right engineering line, because the
  measurement does not exist at any resolution.
* **Кazakhstan ЗРК «О персональных данных и их защите» (94-V)** — attaching a child's name
  to behavioural observations is processing personal data and needs the school's written
  decision, a retention period, and an enforcement mechanism. `INTEGRATION.md` §10.
* **Ranking or scoring children against each other** is the thing to avoid on both legal
  and engineering grounds: an ordering of the activity index is substantially an ordering
  of how well the camera sees each seat. `metrics/trend.py` compares a pupil only with
  themselves and emits no flag, no rank and no sort order.

**Nothing in this section is legal advice.** It is a list of the questions a lawyer should
be asked, written down so that they are asked before a school signs anything.

---

## Tailwind CSS — MIT (not a blocker, and it does travel)

Used by: nothing at runtime. It is a BUILD tool — `qorgan-ai-main/package.json` compiles
`src/qorgan/web/tailwind.css` into `static/app.css`, and that generated file is what
`cabinet/skin.py` vendors into this package's static export.

So no Tailwind code ships, but Tailwind's OUTPUT does, and its MIT attribution banner
travels at the top of it:

    /*! tailwindcss v4.3.3 | MIT License | https://tailwindcss.com */

**That banner must stay.** MIT requires the notice to survive in copies, and this is the
notice. `tests/test_cabinet.py::test_every_page_is_self_contained` was tightened rather
than allowed to fail on it: a URL inside a CSS comment is a licence notice, not something a
browser fetches, and the test now checks what the page would actually REQUEST. An offline
export is still offline.

Nothing about this is a delivery blocker. It is recorded here because a generated file that
carries somebody else's licence text is exactly the kind of thing nobody remembers to check
before shipping.
