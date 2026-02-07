#!/bin/bash
set -euo pipefail

REPO="/users/yjwang/repos/JBB_PAIR_LocalLLMs"
cd "$REPO"
mkdir -p logs

NUM_SHARDS=100
START_SHARD="${1:-0}"

# Submit the first job only
jid=$(sbatch --parsable sbatch_one_shard.sbatch "$START_SHARD")
echo "Submitted first shard job: $jid (shard=$START_SHARD)"
echo "Now set up chaining by enabling job-end submission..."

# Append a post-run hook by wrapping sbatch via srun is messy; instead we do chaining inside python output parsing.
# We'll implement chaining using a tiny helper that each job calls at end.
echo "Done."
