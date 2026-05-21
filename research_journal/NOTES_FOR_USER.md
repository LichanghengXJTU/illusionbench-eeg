# Notes for user

Append-only async channel from the autonomous research loop to the user. The loop
writes here when it discovers a NEED that requires user action (new resource,
paid access, lab data, etc.). The loop does NOT stop — it picks the next-best
action and keeps going.

User: to resolve a NEED, edit this file and add a `RESOLVED:` line under that
entry (or commit/push from your phone via GitHub). The loop scans this file each
tick.

---

(none yet)
