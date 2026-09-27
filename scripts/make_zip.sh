#!/bin/bash
# Builds <team>_submission.zip in the layout the organisers require.
#   scripts/make_zip.sh <team_name> <submission_dir with matching_results.tsv + candidate_pairs.tsv>
set -euo pipefail
TEAM=$1; SUB=$2
STAGE=$(mktemp -d)/pkg
mkdir -p $STAGE/output $STAGE/code
cp $SUB/matching_results.tsv $SUB/candidate_pairs.tsv $STAGE/output/
rsync -a --exclude '__pycache__' --exclude '.DS_Store' code/business_entity_resolution $STAGE/code/
cp Documentation_template.md $STAGE/
python3 utils/validate_submission.py --matching $STAGE/output/matching_results.tsv --candidate $STAGE/output/candidate_pairs.tsv --test-dir dataset/test | tail -1
(cd $STAGE && zip -qr "$OLDPWD/${TEAM}_submission.zip" .)
unzip -l "${TEAM}_submission.zip" | tail -3
