"""What a role may do, as a set of capabilities rather than a rank.

This replaced a strict total order -- operator implied admin implied developer -- under
which every gated route required exactly OPERATOR. That ordering violated the school's
§14 by construction. To let a canteen worker open the canteen journal you had to grant
OPERATOR, and OPERATOR, by the ordering, also opened the bullying event log, the review
controls, the live corridor previews, and /media: snapshots and clips of children. §14
says canteen staff work "БЕЗ доступа к буллингу". A rank cannot say that, because every
role able to see the canteen sat at or above the role that sees everything.

The roles §13/§14 describe overlap, they do not nest: a psychologist reads confidential
notes but not admin settings; a canteen worker sees the canteen but not bullying; an
operator sees events but not the psychologist's notes. Sets express that; an ordering
cannot express it at all.

So: capabilities are granted, never inherited. Adding a capability to a role grants that
one capability and nothing else, and the table below is the whole of the policy -- if a
role is not written here, it can do nothing.

Every capability names routes that exist today. Capabilities for features that do not
exist yet are deliberately absent: a permission guarding nothing is a guess, and it would
be enforced by nobody. The pupil registry was on that list until it was built;
`VIEW_PUPILS` and `MERGE_PERSONS` were added WITH the routes they guard, in the same
change, which is the only order that keeps this rule true. **The psychologist's notes were
the standing example of the rule, cited in this docstring and in the note on
`VIEW_LESSON_METRICS` below, and on 2026-07-28 they stopped being an example**: §13's pages
exist, so the three capabilities that guard them and the role that holds them arrive in the
same change as the routes. Both notes have been rewritten rather than deleted — they were
correct while they stood, and what changed is the world, not the rule.
"""

from __future__ import annotations

from enum import StrEnum

from qorgan.enums import UserRole


class Capability(StrEnum):
    """One capability per thing a person can actually reach."""

    VIEW_CAMERAS = "view_cameras"  # `/`, /api/cameras, /preview/{camera}.jpg
    # /notifications is the same disclosure as /events told from the delivery side -- the
    # summary, the camera and the minute of a bullying incident -- so it is the same grant
    # rather than a second one. web/routes/notifications.py records why a separate
    # capability was rejected: it would be a second door into §14, openable by a role
    # explicitly denied the first.
    VIEW_BULLYING = "view_bullying"  # /events, /notifications
    REVIEW_BULLYING = "review_bullying"  # /events/{id}/review
    VIEW_CANTEEN = "view_canteen"  # /canteen, /canteen/export.csv

    # §12.1's weapon alerts, and the ruling on one. TWO grants over one page, for the
    # same reason VIEW_BULLYING and REVIEW_BULLYING are two: reading a log and acting on
    # it are not one right.
    #
    # And here the second one is not a quality signal, it is THE PRODUCT. A weapon alert
    # is never auto-actioned (§12.1, `docs/questions-for-school.md` §7): the system's
    # whole output is a question, and `CONFIRM_WEAPON_ALERT` is the right to answer it
    # with your name attached. Whoever holds it decides whether a school treats a child
    # as armed. That is not a permission to bundle with "may look at a page", and this is
    # the one place in this enum where the split has consequences outside a browser.
    #
    # Both arrive WITH `/weapons` in the same change, which is the only order that keeps
    # this module's closing rule true.
    VIEW_WEAPONS = "view_weapons"  # /weapons
    CONFIRM_WEAPON_ALERT = "confirm_weapon_alert"  # /weapons/{id}/rule

    # /media is three kinds of thing behind one URL, and it used to be one grant
    # (`VIEW_MEDIA`). The tree is mixed: `snapshots/` and `clips/` are the evidence for one
    # incident involving a few named children; `people/` is a photographic register of
    # EVERY pupil in the school. They answer different questions and they are needed by
    # different people, so they are granted separately -- see `web.routes.media`.
    VIEW_BULLYING_MEDIA = "view_bullying_media"  # /media/snapshots/…, /media/clips/…
    VIEW_PUPIL_PHOTOS = "view_pupil_photos"  # /media/people/…

    # /logs -- the diagnostics page. Deliberately NOT folded into VIEW_CAMERAS even
    # though the camera wall already shows a worker table: the wall shows a bounded
    # summary, and /logs shows the log records themselves, which quote exception strings,
    # file paths, RTSP hosts and whatever the emitting module put in them. §14 gives the
    # оператор безопасности "просмотр тревог; подтверждение/отклонение событий; просмотр
    # клипов" -- the server is not on that list. "Управление серверами" is the
    # суперадминистратор's, and `UserRole.DEVELOPER` is already documented as the debug
    # views. Those two, and no one else.
    VIEW_DIAGNOSTICS = "view_diagnostics"  # /logs

    # /settings: the whole installation, read-only -- every camera's RTSP host, every
    # detector threshold, every ROI rectangle, and the file each value comes from.
    #
    # NOT bundled with VIEW_CAMERAS. Watching a corridor and reading how the detector is
    # tuned are different jobs, and this page answers a question about the SYSTEM rather
    # than about a child. Held by ADMIN and DEVELOPER, whose §13 descriptions are exactly
    # "настройки" and "ROI calibration".
    #
    # There is deliberately no companion capability for CHANGING any of it: nothing on that
    # page writes, because there is no control channel from the web process to the
    # supervisor and a value that lives in YAML must not also live in a browser form. The
    # rule at the foot of this module applies -- a permission guarding nothing is a guess.
    # When such a channel exists, the right to use it is a NEW capability and never this
    # one: reading a threshold is not the same right as switching a school's cameras off.
    VIEW_SETTINGS = "view_settings"  # /settings

    # Backups are TWO grants over one page, for the same reason /media became two over one
    # tree: reading and doing are different questions asked by different people.
    #
    # `VIEW_BACKUPS` answers "has a copy of the database been taken, and when?" -- a
    # question a headteacher is entitled to ask, whose answer changes nothing. It is also
    # the only way anyone finds out that the scheduled task has been failing since March.
    #
    # `CREATE_BACKUP` writes a new file that contains EVERY child in the school -- faces,
    # meals, incidents -- onto the same disk the recordings are written to. It is an
    # action with two costs (a second copy of the children's data exists; the disk gets
    # smaller), and neither is undone by closing the page.
    #
    # One grant could not express "may check, may not take", so the person who needs the
    # answer would have to be handed the button. Kept apart here in the same way
    # VIEW_BULLYING and REVIEW_BULLYING are: reading a log and acting on it are not one
    # right, and the model has to be able to say so before anybody needs it to.
    VIEW_BACKUPS = "view_backups"  # GET /backups
    CREATE_BACKUP = "create_backup"  # POST /backups

    # The registry itself: who is enrolled, what class they are in, and whether the system
    # can recognise them at all. Separate from VIEW_PUPIL_PHOTOS for the same reason that
    # capability was split off VIEW_MEDIA -- "who is on the roster" and "what does this
    # child look like" are different questions, and a page that answers the first does not
    # need to answer the second. The registry page carries no photograph; the duplicate
    # page does, and asks for both.
    VIEW_PUPILS = "view_pupils"  # /pupils, /pupils/{id}/canteen, /pupils/duplicates

    # The lesson reports (§12.4, cut back to what §8 promised): per-lesson counts of
    # raised hands, standing up and time away from a place, **by anonymous track**.
    #
    # Added WITH `/lessons` and `/lessons/{id}` in the same change, which is the only
    # order that keeps this module's closing rule true -- and it is granted to exactly ONE
    # role, which took more thought than the grant looks.
    #
    #   * ADMIN, because §14 gives the администратор школы "отчеты" and lists them under
    #     nobody else. A lesson report is a report about the school's own children,
    #     produced for the school to act on or ignore.
    #   * NOT the OPERATOR. §14 gives them "просмотр тревог; подтверждение/отклонение
    #     событий; просмотр клипов". A lesson report is none of those, and an operator
    #     who cannot open it is not thereby prevented from doing their job -- which is
    #     the test `_MAINTENANCE_CAPABILITIES` is held to above.
    #   * NOT the DEVELOPER, and this is the deliberate one: it is the first
    #     child-facing capability ADMIN holds and DEVELOPER does not. Everything the
    #     developer login holds is either the INSTALLATION (/logs, /settings, /backups)
    #     or something an operator already saw. These numbers are new, they are about
    #     children, and **not one threshold behind them has been validated against a real
    #     lesson** -- so the smallest possible readership is the right one until they have.
    #     A developer who needs them gets the grant from an admin who chose to give it.
    #   * The PSYCHOLOGIST, added 2026-07-28 with §13's pages. This bullet used to read
    #     "NOT a psychologist role, which is not created here and must not be", and the
    #     reason it gave was that `qorgan.classroom` produces no signal and no ranking, so
    #     there was no psychologist's page for the role to open. That reason has expired,
    #     not been overruled: the pages exist now. The other half of it stands and is why
    #     the grant is safe rather than merely convenient — the lesson report is per
    #     ANONYMOUS TRACK and carries no `person_id`, so however long a psychologist reads
    #     it, it cannot become a claim about a named child. §13 asks them to see behaviour
    #     over time and this is the only place that data exists; what it can honestly tell
    #     them about one pupil is nothing, and `LessonReport.caveat()` says so on the page.
    VIEW_LESSON_METRICS = "view_lesson_metrics"  # /lessons, /lessons/{id}

    # The IMPORTED offline classroom analyses: the cabinet's classroom pages, built over
    # `db/models/classvision.py` in this same change. The exact URLs are the routes' own
    # business and are deliberately not quoted here -- a path written in this comment is a
    # claim nothing checks, and this file already carries four of those as a warning.
    #
    # **Deliberately NOT folded into `VIEW_LESSON_METRICS`.** That capability guards the LIVE
    # worker's per-ANONYMOUS-TRACK counts, and the note above it says in as many words why
    # the psychologist's grant is safe: those rows carry no `person_id`, so however long
    # anybody reads them they cannot become a claim about a named child. **These rows can.**
    # A place whose seating plan a class teacher has attested carries `person_id`, so this is
    # a different disclosure and gets its own grant rather than riding on a sentence that is
    # true of the other table.
    #
    # Granted to ADMIN and PSYCHOLOGIST. **NOT the DEVELOPER**, following the same line
    # `VIEW_LESSON_METRICS` drew: the supplier's debug login holds the INSTALLATION, and
    # longitudinal observations about children are not the installation. **NOT the OPERATOR**
    # — §14 gives them alarms, verdicts and clips; a term's history of one place is none of
    # those. NOT CANTEEN_STAFF, NOT SUPERADMIN, neither of which holds any child-facing
    # capability at all.
    VIEW_CLASSROOM_ANALYSIS = "view_classroom_analysis"  # the imported-analyses pages

    # §13's cabinet: incidents a named person referred, and one pupil's canteen attendance
    # over time. Added WITH `/psychologist` and `/psychologist/pupils/{id}`.
    #
    # It is NOT `VIEW_BULLYING` narrowed. The events log is every candidate the detector
    # produced, false positives included; this is the short list somebody decided to hand
    # on, which is what §14 gives the психолог («подтвержденные случаи»), and it is a
    # smaller disclosure reached through a different door rather than the same door with a
    # filter on it. A psychologist holding VIEW_BULLYING would read the whole corridor log.
    VIEW_PSYCHOLOGIST_CABINET = "view_psychologist_cabinet"  # /psychologist, …/pupils/{id}

    # **The confidentiality boundary §13 states outright**: «обычный оператор не должен
    # видеть конфиденциальные записи психолога». Two grants, reading and writing, for the
    # reason the backups page is two: they are different acts by potentially different
    # people, and one grant could not express "may read the history, may not add to it".
    #
    # THE ONLY CAPABILITIES IN THIS ENUM THAT `UserRole.ADMIN` DOES NOT HOLD. That is a
    # deliberate default and not a wall — MANAGE_USERS can mint a psychologist account, so
    # an administrator who wants these can have them. The difference is that minting the
    # account is a visible act somebody can later read in the log, and quietly opening a
    # colleague's confidential file is not. The default is where the school's expectation
    # lives; the wall is not something a permission table can build.
    VIEW_PSYCHOLOGIST_NOTES = "view_psychologist_notes"  # GET /psychologist/notes/{id}
    WRITE_PSYCHOLOGIST_NOTES = "write_psychologist_notes"  # POST /psychologist/notes/{id}

    # §9: «Сотрудник должен иметь возможность отметить … передано психологу». A HUMAN
    # referring a child is the product; the SYSTEM referring one is what §8 promised the
    # school would never happen, so this is a right held by people and by nothing else.
    #
    # Separate from REVIEW_BULLYING, which it would have been easy to fold into: a verdict
    # says "this was, or was not, an assault", and a referral hands a named child to a
    # specialist. Different acts, different consequences, and the second one outlives the
    # first — see `events.referred_at`. `/events/{id}/review` REFUSES the referral verdict
    # precisely so that this grant is the only way the mark can be made, and the mark
    # therefore always has a name and a time beside it.
    REFER_TO_PSYCHOLOGIST = "refer_to_psychologist"  # POST /events/{id}/refer

    # **Merging is a mutation of a child's identity, so it is a right of its own.**
    #
    # Every capability above answers "may this person LOOK at X". This one answers "may
    # this person DECIDE that two school ids are one human", and the two must never be the
    # same grant, because the consequences are not the same kind of thing:
    #
    #   * A merge re-points photographs, embeddings and CANTEEN SESSIONS onto one id and
    #     retires the other. The retired id leaves the gallery entirely.
    #   * One of this school's six duplicate pairs crosses the pupil/staff line
    #     (`student_470 / staff_334`). Staff never open a meal session, so on that pair the
    #     merge decides whether a CHILD IS FED -- and if it goes the wrong way nothing
    #     reports it, because the number that would have said so is the number that stops
    #     being produced (`identity/merge.py`).
    #   * `7-А 438/439` may be identical twins. Getting that wrong invents an identity.
    #
    # Reading the registry is how you PREPARE that decision, and the person who prepares it
    # is not always the person entitled to make it. Folding the two together would mean
    # that granting anybody the ability to check who is enrolled twice also granted them
    # the ability to erase one of the two, which is the §14 mistake one more layer down:
    # a grant shaped so that needing part of it takes all of it.
    MERGE_PERSONS = "merge_persons"  # POST /pupils/duplicates/merge, …/undo

    # NOT ANOTHER VIEW. Every capability above lets somebody SEE something; this one lets
    # them decide who sees it. Whoever holds it can create an account with `role=admin`,
    # including for themselves, and an admin holds every capability in this enum except the
    # psychologist's notes -- the live corridor cameras, the bullying log, the photograph of
    # every pupil in the school. So this is the one grant where being wrong is not a page
    # leaking: it is the whole table above becoming advisory. (And the exception is not
    # cover: the same grant mints a psychologist account, which is exactly why the notes
    # are held back by default rather than by a wall -- see VIEW_PSYCHOLOGIST_NOTES.)
    #
    # It is a capability rather than a role check for the same reason as everything else
    # here (see the module docstring), but it is the capability whose grant list is
    # shortest on purpose -- see ROLE_CAPABILITIES below.
    MANAGE_USERS = "manage_users"  # /users, /users/{id}/role, /users/{id}/active

    # §14's "управление школами": the register of schools on this installation, and the
    # right to add one or rename one. Added WITH `/schools` in the same change, which is
    # the only order that keeps this module's closing rule true.
    #
    # **It is a capability over the TENANCY, not over any tenant.** The page shows how
    # many pupils, cameras and accounts each school has -- numbers about the installation
    # -- and no route it opens returns a child's name, photograph or incident. That is
    # deliberate and it is the whole shape of this grant: a superadmin exists so a school
    # does not have to phone the vendor to get its own login, not so the vendor can read
    # twenty schools' corridors. Every child-facing capability stays where §14 put it,
    # inside the school, held by people the school employs.
    #
    # It is held by `SUPERADMIN` alone. Granting it to an ADMIN would let one school
    # rename another; granting it to DEVELOPER would recreate "the supplier can always let
    # themselves in", which the audit condemned and MANAGE_USERS is already kept from.
    MANAGE_SCHOOLS = "manage_schools"  # /schools, /schools/new, /schools/{id}/name


# What an operator does today. Named once so that ADMIN and DEVELOPER cannot drift away
# from it silently: under the ordering this replaced, both reached every operator route,
# and the school's headteacher noticing on Monday that their account lost the events page
# is not an acceptable way to discover a refactor.
_OPERATOR_CAPABILITIES = frozenset(
    {
        Capability.VIEW_CAMERAS,
        Capability.VIEW_BULLYING,
        Capability.REVIEW_BULLYING,
        # Both halves of the old single VIEW_MEDIA. An operator reached the whole tree
        # before the split and reaches the whole tree after it: this is a split, not a cut.
        Capability.VIEW_BULLYING_MEDIA,
        Capability.VIEW_PUPIL_PHOTOS,
        Capability.VIEW_CANTEEN,
        # An operator already reached every pupil photograph under the old whole-tree
        # grant, so being able to read the register those photographs belong to is not a
        # widening. MERGE_PERSONS is not here, and that is the point of it.
        Capability.VIEW_PUPILS,
        # §14 gives the оператор безопасности "просмотр тревог; подтверждение/отклонение
        # событий" -- and a weapon alert is a тревога that exists ONLY to be confirmed or
        # rejected by a person. An operator who cannot rule on one is an operator watching
        # a page whose single control they may not press, which leaves §12.1's human
        # confirmation with nobody to make it.
        #
        # Put in the OPERATOR set rather than granted separately so that VIEW_WEAPONS and
        # /notifications cannot drift apart: a weapon alert raises a Telegram, so its
        # summary already appears on /notifications, which is VIEW_BULLYING. Anybody who
        # can read the notification can read the alert it came from, by construction, and
        # there is no arrangement of these two grants that leaks one without the other.
        Capability.VIEW_WEAPONS,
        Capability.CONFIRM_WEAPON_ALERT,
    }
)

# Looking after the machine, as opposed to looking after the children on it. Nothing here
# is part of watching a corridor or serving lunch, and an operator who has neither is not
# thereby prevented from doing their job -- which is the test for whether a capability
# belongs in the operator set.
_MAINTENANCE_CAPABILITIES = frozenset({Capability.VIEW_BACKUPS, Capability.CREATE_BACKUP})

ROLE_CAPABILITIES: dict[UserRole, frozenset[Capability]] = {
    # §9 gives the referral to «сотрудник», and the operator is the member of staff sitting
    # in front of the events log. It is granted HERE rather than inside
    # `_OPERATOR_CAPABILITIES` on purpose: that set flows into DEVELOPER, and referring a
    # named child to the school psychologist is a claim the SCHOOL makes about a person --
    # the same line MERGE_PERSONS and MANAGE_USERS are on, and the supplier's debug login
    # does not stand on it. The divergence is written out so it is read, not inherited.
    UserRole.OPERATOR: _OPERATOR_CAPABILITIES | {Capability.REFER_TO_PSYCHOLOGIST},
    # ADMIN and DEVELOPER hold everything about the INSTALLATION -- /logs, /settings,
    # /backups -- because §13 makes both responsible for running and maintaining it.
    #
    # THEY DIVERGE ON TWO GRANTS, AND BOTH ARE ADMIN'S ALONE. MERGE_PERSONS decides that
    # two school ids are one child; MANAGE_USERS decides who may log in at all. Each is a
    # claim the SCHOOL makes about people, and the account that speaks for the school is
    # the headteacher's, not the vendor's. A developer login exists to debug the system:
    # granting it either one would let the supplier rewrite a child's identity, or mint
    # themselves an admin at any hour -- and an admin reaches live video of children and
    # every pupil's photograph. "The supplier can always let themselves in" is the
    # arrangement the audit condemned. A developer who needs an account gets one from an
    # admin who chose to give it, like anybody else.
    #
    # EVERY grant is named in one expression ON PURPOSE. These arrived on five branches
    # that each rewrote these two lines, and taking any single version alone would have
    # silently REVOKED the others' pages -- with a green suite, because each branch's tests
    # assert only their own capability. Add to the union; never replace it.
    UserRole.ADMIN: _OPERATOR_CAPABILITIES
    | _MAINTENANCE_CAPABILITIES
    | {
        Capability.VIEW_DIAGNOSTICS,
        Capability.VIEW_SETTINGS,
        Capability.MERGE_PERSONS,
        Capability.MANAGE_USERS,
        # §14 lists "отчеты" under the администратор школы and under no other role. This
        # is the first capability about CHILDREN that ADMIN holds and DEVELOPER does not;
        # see the note on VIEW_LESSON_METRICS for why the readership starts this small.
        Capability.VIEW_LESSON_METRICS,
        # The imported classroom analyses, on the same §14 «отчеты» argument, and with the
        # same readership as the live lesson report plus nobody: an admin who can read the
        # anonymous per-track counts can read the per-place ones, and these additionally
        # carry a name wherever a seating plan was signed.
        Capability.VIEW_CLASSROOM_ANALYSIS,
        # §9's mark, so the headteacher can make it too. The DEVELOPER does not get it --
        # see the note on the OPERATOR entry above.
        Capability.REFER_TO_PSYCHOLOGIST,
        # The cabinet, because somebody at the school other than the psychologist has to be
        # able to see that referrals are arriving and that the canteen signal is or is not
        # accumulating -- the failure mode this whole module was written against is a page
        # that LOOKS like it is working. The two NOTES capabilities are deliberately NOT
        # here, and they are the only two in the enum an admin does not hold.
        Capability.VIEW_PSYCHOLOGIST_CABINET,
    },
    UserRole.DEVELOPER: _OPERATOR_CAPABILITIES
    | _MAINTENANCE_CAPABILITIES
    | {Capability.VIEW_DIAGNOSTICS, Capability.VIEW_SETTINGS},
    # §14, and the reason this module exists. The canteen journal, and nothing else: no
    # events, no review, no previews, and neither half of the media tree.
    UserRole.CANTEEN_STAFF: frozenset({Capability.VIEW_CANTEEN}),
    # §14: «сигналы по ученикам; подтвержденные случаи; история наблюдений; свои
    # комментарии». Written out in full rather than built from another role's set, because
    # this role overlaps every other one and nests inside none of them -- which is the
    # whole argument of this module's docstring, arriving in a role for the first time.
    #
    # NOT here, and each absence is a decision:
    #   * VIEW_BULLYING / REVIEW_BULLYING. §13 says «психолог не должен видеть все
    #     административные настройки», and §14 gives them «подтвержденные случаи», not the
    #     raw candidate log with its false positives. What was handed to them is on
    #     /psychologist; what the detector merely suspected is not their work.
    #   * REFER_TO_PSYCHOLOGIST. They are the recipient of a referral, not its author.
    #   * Every installation capability -- /logs, /settings, /backups -- which is the
    #     sentence §13 spends its second paragraph on.
    #
    # VIEW_PUPILS and VIEW_CANTEEN ARE here, and this is the grant worth arguing about.
    # §13 lists «посещаемость» for the psychologist, and in this system attendance IS the
    # canteen record -- there is no other attendance signal, and the classroom half is
    # anonymous by construction. Granting them means the per-pupil trend page can require a
    # SUPERSET of what `/pupils/{id}/canteen` already requires, so it opens no door to
    # anybody who did not already hold one: the second-door mistake this module exists to
    # prevent (see the note on VIEW_BULLYING) is avoided by widening the ROLE rather than
    # by narrowing the page. The cost is real and is not hidden: it also hands them
    # /canteen and its CSV export, which is the whole school's day rather than one child's
    # term, and that is more than §13 asked for. Splitting "one named pupil's meal record"
    # out of VIEW_CANTEEN is the fix if the school wants one; it is a question for them,
    # and `docs/questions-for-school.md` §10 is where it is asked.
    UserRole.PSYCHOLOGIST: frozenset(
        {
            Capability.VIEW_PSYCHOLOGIST_CABINET,
            Capability.VIEW_PSYCHOLOGIST_NOTES,
            Capability.WRITE_PSYCHOLOGIST_NOTES,
            Capability.VIEW_PUPILS,
            Capability.VIEW_CANTEEN,
            Capability.VIEW_LESSON_METRICS,
            # §13 asks the psychologist to see behaviour over time, and until this change the
            # only classroom data that existed was per anonymous track — «what it can honestly
            # tell them about one pupil is nothing». The imported analyses accumulate per
            # PLACE, which is the first classroom signal that survives a term, so this is the
            # grant that makes that sentence of §13 answerable. It stays honest by
            # construction: with no signed seating plan the pages say «место 3», and the name
            # arrives only from `classvision_attestations`.
            Capability.VIEW_CLASSROOM_ANALYSIS,
            # **The grant `VIEW_MEDIA` was split apart to make possible**, now actually
            # made. `web/routes/media.py` and `tests/test_web_media_capabilities.py` both
            # name this exact person as the reason: "the psychologist §13 describes is
            # asked about one incident and needs its clip; nothing in that work touches the
            # enrolment gallery". So the incident half, and not `VIEW_PUPIL_PHOTOS` -- a
            # referral with the evidence stripped out is a sentence and a child's name.
            #
            # It does reach snapshots and clips of incidents nobody referred, because the
            # media tree is not indexed per referral. That is a real widening and the
            # narrower alternative (a per-event grant) does not exist in this model; the
            # paths are timestamped rather than enumerable, and /events itself stays shut.
            Capability.VIEW_BULLYING_MEDIA,
        }
    ),
    # §14: "управление школами, серверами". TWO grants, and the interesting part is
    # everything that is NOT here.
    #
    # `MANAGE_SCHOOLS` is the schools register. `VIEW_DIAGNOSTICS` is `/logs`, and it is
    # granted because the note on that capability above already said, before this role
    # existed, that "управление серверами" is the суперадминистратор's -- it was held by
    # ADMIN and DEVELOPER only because there was nobody else to hold it.
    #
    # There is no VIEW_CAMERAS, no VIEW_BULLYING, no VIEW_PUPILS and neither half of
    # /media. **This is the only role on the installation that could reach every school at
    # once, so it is the role that must reach the fewest children.** A superadmin who
    # needs to see a school's corridor asks that school for an account, like anybody else;
    # the same rule that keeps MERGE_PERSONS and MANAGE_USERS away from the developer
    # login, one layer up. It is also why `users.school_id` is NULL for exactly this role
    # and no other: a row that belongs to no school cannot be filtered into one.
    UserRole.SUPERADMIN: frozenset(
        {Capability.MANAGE_SCHOOLS, Capability.VIEW_DIAGNOSTICS}
    ),
}


def capabilities_for(role: UserRole) -> frozenset[Capability]:
    """What this role may do. A role nobody wrote down may do nothing.

    Deny by default, the same rule the auth middleware follows: a role missing from the
    table is a bug, and a bug in a permission table must fail shut. `test_every_role_
    states_its_capabilities_in_writing` is what actually catches the omission.
    """
    return ROLE_CAPABILITIES.get(role, frozenset())
