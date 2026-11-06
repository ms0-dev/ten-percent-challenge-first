# The 10% challenge

> **Solution in this fork:** [`solution/PROOF.md`](solution/PROOF.md) - 10.48%
> of Qwen3-30B on disk, locked exam 96.00% vs base 95.80%, training-free
> growth. Summary: [`solution/README.md`](solution/README.md).

![The 10% challenge](challenge.png)

Small, just as good, and still learning.

## The challenge

1. **Start from one trained model.** It already does the job, and its score is the bar to meet.
2. **Keep about 10% of its size.** You can't train the weights. Every other technique is fair game.
3. **Match the base.** Your model gets the same score as the one you started from.
4. **Pass an exam it never saw.** Set aside a held-out, unseen subset before you begin. Keep it locked away until the very end, and never use it to build or to choose.
5. **Let it grow with user feedback.** It learns the new and keeps the old, still with no weight training, and it still passes the locked exam.

Submit your proof by [opening a GitHub issue](https://github.com/DeusExperiments/ten-percent-challenge/issues/new) on this repo. Only the first person to submit a complete, working proof of beating the challenge qualifies.

How would you do it?
