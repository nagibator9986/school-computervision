"""What is wrong right now, and what went wrong recently.

The school asked for five things in one line (client §9): "**Logs**: ошибки камер; ошибки
моделей; ошибки базы; причины пропуска тревог; перезапуски." They do not all live in the
same place, and pretending they did is what this package exists to avoid.

**Two of them are TABLES, and are read from the tables.**

  * *Перезапуски* -- `worker_heartbeats.restart_count`, bumped by the supervisor and by
    nothing else (`supervisor._supervise`). Counting restarts by grepping log text would
    disagree with that column the moment a line rotated off the end of a file, and a
    restart count that is quietly low is exactly the "everything looks healthy" the
    supervisor was written to end.
  * *Причины пропуска тревог* -- the `notifications` rows. `status`, `attempts` and
    `last_error` are written by `notify.queue` on every attempt, so the answer to "why did
    that alert not arrive" is a row, not a sentence somebody has to find.

**Three of them are not tables at all.** Camera connection errors, model failures and
database errors have no table anywhere in this system. `worker_heartbeats.last_error` is
one string per worker group, overwritten by the next heartbeat five seconds later: it is
the CURRENT state, and rendering it under a heading like "recent errors" would be a
number that looks true and is not. The rotating files written by `logging_setup` are the
only historical record there is, so `logfiles` reads those -- and says out loud how far
back it actually looked, because a horizon nobody states is a horizon everybody
misreads.

**Nothing here writes.** These functions are reached from a GET. The legacy restarted the
AI workers when somebody opened a tab, with a five-second `thread.join()` inside the HTTP
handler; no function in this package rotates a file, creates a directory, sweeps a table
or pokes a worker.
"""
