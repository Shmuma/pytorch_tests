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

Item 2 don't have much progress, stopped.

Next experiment will be to disable epsilon-greedy. So, I have those versions running:

1. Oct08_13-39-27_home-lr=0.001 -- no clipping, small LR
2. Oct08_15-15-04_gpu-lr=0.001-clp=100 -- clipping 100, small LR, almost the same progress as 1
3. Oct08_17-00-31_gpu-lr=0.001-clp=100 -- the same, no epsilon greedy

Without epsilon-greedy dynamics is the same as with it.

The next two experiments will be:
* to disable updates with too short value steps.
* decrease LR even more, to 0.0001

Running experiments:
1. Oct08_18-23-34_gpu-no-short-rw -- the same params, no short reward updates
2. Oct08_18-24-54_gpu-no-short-rw-lr=1e-5 -- the same, lr=1e-5

Version with 1e-5 earned 6000 scores, but learned it back. What is strange -- value loss is huge and improves very slowly.

I think key here is to make this value learned quickly.

Maybe, prioritized experience replay will help here a lot.

First version was stopped and started version with increased batch size (4 -> 64)
This version is Oct08_20-18-21_gpu-lr=1e-5-batch=64

# 2017-10-09

Last two variants (with LR=1e-5 and batch=64) are good, batch=64 is more or less stable and 
could achieve 12000 score (TODO: check state of the art).

Next steps will be to implement periodical test on frozen network and do saving of the best model.