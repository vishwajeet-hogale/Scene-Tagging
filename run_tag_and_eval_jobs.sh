#!/bin/bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"

cd "$repo_dir"

tag_job_id="$(sbatch --parsable tag_scenarios.sbatch)"
eval_job_id="$(sbatch --parsable --dependency=afterok:${tag_job_id} eval_scenarios.sbatch)"
analyze_job_id="$(sbatch --parsable --dependency=afterok:${tag_job_id} analyze_taxonomy.sbatch)"

echo "Submitted tag job: ${tag_job_id}"
echo "Submitted eval job: ${eval_job_id}"
echo "Submitted analyze taxonomy job: ${analyze_job_id}"
echo "Eval job will start after tag job completes successfully."
echo "Analyze taxonomy job will start after tag job completes successfully."