"""Classroom lesson metrics: physical, observable facts about anonymous tracks.

What this package may report is fixed by what the school was promised in writing
(`docs/questions-for-school.md` §8), and it is a SHORT list: how many times a hand went
up, how many times someone stood up, how long they were away from their place, and
whether they were at the lesson at all.

**What it must never report, and why the refusals are in the code rather than in a
document.** The school was told, in writing:

  * **No conclusion about emotion, attention or engagement from a face.** «Это не
    работающая наука, и в школе ей не место». There is no expression classifier here and
    there is no place to add one: nothing in this package reads a face at all.

  * **No diagnosis, and no referral to a psychologist from the system.** The report is
    counts with units. Every verb in it is one a camera can witness. There is no score,
    no ranking, no "signal", and no threshold above which a child is flagged — because
    the decision belongs to a person and a number that fires is a decision made.

  * **No judgement about a teacher's work.** Nothing here measures the adult in the room.
    «Сидит» is not «не ведёт урок», and this package is not equipped to tell the
    difference, so it does not try. §12.5 is deliberately not built (see below).

  * **No identity recognition in a classroom.** Metrics are per TRACK and anonymous. This
    is not squeamishness, it is a measurement: at corridor distance this school's own
    footage yielded 14 970 faces, median 11.5 px, and **zero** were recognised. A face
    across a classroom is the same size. Attaching a name here would attach it on
    evidence that has been measured to carry none, so nothing in this package writes,
    reads or accepts a `person_id`, and `LessonTrack` has no column for one.

**A track is not a child, and the report says so on every surface.** ByteTrack loses
people behind other people; one child who is occluded and re-acquired becomes two tracks,
and two children who cross can swap. So the count of tracks is neither an upper nor a
lower bound on the count of children, and `LessonReport.caveat()` is what carries that
sentence to the page, exactly as `canteen.reports.DayReport.caveat()` carries the
canteen's. Never render the numbers without it.

**No threshold in this package has been validated against a real lesson.** We hold no
recording of one — `class/` is photographs. Every number in `config/classroom.py` is an
estimate reasoned from body geometry, each labelled as such beside itself. They are
starting points for the day the school sends footage, not tuned values.

--------------------------------------------------------------------------------
**THE PART OF §8 THIS PACKAGE CANNOT DELIVER, AND IT IS NOT AN OVERSIGHT.**

§8 does not only list the four facts above. It also tells the school what a signal would
be: «Сравниваем ребёнка только с ним самим — с его собственной нормой за предыдущие
4 недели. Сигнал — это устойчивое падение ниже его собственной нормы на нескольких
уроках подряд… тихий ребёнок не сработает никогда — сработает только тот, кто изменился.»

**That signal cannot be built on what this package produces, and the obstacle is §8
itself.** A four-week personal baseline requires knowing that the child in today's lesson
is the child from a lesson three weeks ago. That is identification. §8 rules it out inside
a classroom in the same breath — and the school's own footage says it would not work even
if it were permitted: 14 970 corridor faces, median 11.5 px, zero recognised.

A track lives minutes and dies to the first occlusion; it cannot carry a four-week norm.
Nothing here will pretend otherwise, which is why these tables have no `person_id` and no
column a baseline could hide in. So the honest statement is that **§8's two halves are in
tension**: the comparison it promises needs an identity the same paragraph forbids.

What exists here is the half that can be measured honestly — counts, per lesson, per
anonymous track, with every hole they contain counted beside them. Turning those into a
per-child trend is a decision for the SCHOOL, taken knowing the price: either children are
enrolled by some means that is not a classroom camera, or there is no trend. It is not a
decision to be taken by quietly adding a nullable column.
"""
