# Findings (one line each: F-### date run_id [diff:X] result; answers Q-### if any)
F-001 2026-10-21 20261021-153458_sides [diff:A] 10 eps: 6 success, 2 near_miss, 2 fail (60%; model card 65.4%). Side around the T varies by seed (cw-only: 0,2,8; mostly ccw: 3,4,5): no fixed bias.
F-002 2026-10-21 20261021-153458_sides [diff:A] Switches/ep (0.5 rad): fail 4.5, near_miss 2.5, success 1.5; but 5 of 6 successes switch too and the gap shrinks at 0.8 rad. 2 failures only: suggestive; partly answers Q-001.
