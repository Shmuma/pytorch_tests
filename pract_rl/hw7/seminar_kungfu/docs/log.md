# First version

Has the following tweaks (untuned):
1. target network (sync state and policy net every 10 experience item)
2. batched gradient update: every 4 iterations
3. epsilon-greedy action selector (25% of actions are random)
4. gradient clipping is disabled

The best effect was due to epsilon-greedy selector, as previous version hasn't learned at all.
But still, current version entropy value is almost 0, which is strange, maybe it's worth to:
1. increase epsilon to 50% or more
2. increase entropy beta from 0.1 to higher values

Currently running versions:
1. Oct08_11-00-43_home-first -- no target network (no convergence, stopped)
2. Oct08_11-21-59_gpu-first -- with target network every 10 iters
3. Oct08_11-36-22_gpu-clip -- target net + gradient clipping (L2 norm <= 10)
4. Oct08_13-39-27_home-lr=0.001 -- the same, but with LR=0.001 (10 times less than before)

After looking at net with and without clipping, I have impression that clipping is only makes it worse.
Stopped version 3, restarted it without clipping but with lower LR

Version without clipping stagnated with entropy_loss=0.0...
Next try is to leave clip, but increase it to 100.0
