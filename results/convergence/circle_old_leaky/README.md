# convergence / circle_old_leaky

The same cylinder convergence study as `../circle_old/`, run **with no gap
between the train and test blocks** (gap 0, 481/159/160).

Kept deliberately, as the demonstration of the failure it is named for.
Consecutive snapshots are near-identical, so a test block that begins one frame
after the training block ends is largely a copy of it, and every model scores
better than it should. Read it only next to `../circle_old/`.
