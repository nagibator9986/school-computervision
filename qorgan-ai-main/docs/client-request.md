# Что просила школа

The original ask, verbatim, kept because every later document is a *response* to it and
an argument is worth nothing without the thing it is arguing with. Recorded 2026-07-13;
it had been living in a chat window, which is how it nearly got lost.

> 1. Распознавание оружия по камере наблюдения, если распознала — сигнал.
>
> 2. Камера в классе: следит за поведением ученика. Первую неделю был активен, потом
>    что-то потух — если так, то его к психологу. Вторая камера смотрит на учителя:
>    он реально ведёт урок или просто сидит.
>
> 3. У меня модули по буллингу и распознаванию лиц стоят, их нужно просто на финальном
>    этапе настроить — типа чувствительность реакции (существующая модель, просто
>    откалибровать).

## Where each point actually stands

**Point 3 is the one that was wrong, and it is worth being precise about why**, because
it was the school's own summary of the state of the system and it was mistaken in a way
nobody could have seen from the outside.

"Просто откалибровать чувствительность" was not possible. There was nothing to turn. Two
of the quantities being thresholded were computed incorrectly, so the thresholds were not
insensitive — they were meaningless:

- The bullying detector's acceleration was measured in `(px/frame)/s`, a unit that changes
  meaning with the frame rate, and the threshold sat *below the noise floor of the YOLO
  box itself*. An ordinary walk crossed it on 4-22% of frames depending on the camera. It
  was reacting to the bounding box, not to the children.
- Face recognition ranked candidates by *photo*, so `top1` and `top2` were two pictures of
  the same child and the ambiguity gate compared Alice against Alice. It rejected everyone.
  1 816 of 1 820 canteen records came out `NULL`. No value of the threshold could have
  fixed it; the 18 tuned thresholds in the legacy config bought nothing.

Both are fixed (see `qorgan eval noise-floor`, `faces/matching.py`). Calibration is now
*possible*, and it is now *blocked on data*: a recording of a quiet corridor to measure
the real noise floor, labelled clips to build a PR curve, and a class roster with IDs.
That is the school's to supply and no amount of engineering substitutes for it.

**Point 1 is buildable** and is the least entangled of the three: a separate detection
session with its own class map, at a reduced frame rate, raising an alert a human then
confirms. Never an autonomous trigger — real-time weapon detection on CCTV is documented
as an open problem with high false-positive rates, and in a school a false gun alert has
consequences of its own.

**Point 2 is two different requests wearing one coat**, and they need to be separated
before anything is built. See `docs/superpowers/specs/` for the design that resulted.
