#!/usr/bin/env bash
set -euo pipefail

# Medium-quality end-to-end MedGS4D run for an annotated 4D-Lung study.
# The script is intentionally fail-fast and does not use --force.
# Change the run names before repeating the experiment.

REPO=/home/jovyan/shared/mtm_medgs_stack/repo/medgs4d
PYTHON=/home/jovyan/shared/mtm_medgs_stack/envs/medgs-worf/bin/python
MEDGS_REPO=/home/jovyan/shared/mtm_medgs_stack/repo/MedGS

DICOM_DIR=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/dicom_by_series
PREPARED_ROOT=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/prepared
CANONICAL_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/canonical
DYNAMIC_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/medgs4d

PATIENT_ID=117_HM10395
STUDY_UID=1.3.6.1.4.1.14519.5.2.1.6834.5010.378204929111417980831212264180
STUDY_NAME=patient117_20001024_annotated

CANONICAL_PHASE=0
CANONICAL_ITERATIONS=15000
DYNAMIC_ITERATIONS=7000

CANONICAL_RUN_NAME=phase00_denoised_iter15000
DYNAMIC_RUN_NAME=phase00_canon15000_denoised_holdout_iter7000

STUDY_DIR="$PREPARED_ROOT/$STUDY_NAME"
CANONICAL_RUN="$CANONICAL_ROOT/$STUDY_NAME/$CANONICAL_RUN_NAME"
CANONICAL_CHECKPOINT="$CANONICAL_RUN/model/chkpnt${CANONICAL_ITERATIONS}.pth"
RUN_DIR="$DYNAMIC_ROOT/$STUDY_NAME/$DYNAMIC_RUN_NAME"
FINAL_DYNAMIC_CHECKPOINT="$RUN_DIR/checkpoints/deformation_iter_$(printf '%06d' "$DYNAMIC_ITERATIONS").pth"

cd "$REPO"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
LOG_FILE="$REPO/logs/${STUDY_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log: $LOG_FILE"
echo "Study: $STUDY_NAME"
echo "Study UID: $STUDY_UID"
echo "Canonical iterations: $CANONICAL_ITERATIONS"
echo "Dynamic iterations: $DYNAMIC_ITERATIONS"

# 1. Basic code and data checks.
"$PYTHON" -m compileall -q medgs4d scripts

"$PYTHON" scripts/data.py inspect \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID"

RTSTRUCT_REPORT="$REPO/logs/${STUDY_NAME}_rtstructs.txt"
"$PYTHON" scripts/data.py list-rtstructs \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --referenced-study-uid "$STUDY_UID" \
  | tee "$RTSTRUCT_REPORT"

echo "RTSTRUCT inventory: $RTSTRUCT_REPORT"

# 2. Prepare raw and denoised 4D-CT volumes.
"$PYTHON" scripts/data.py prepare \
  --dicom-dir "$DICOM_DIR" \
  --prepared-root "$PREPARED_ROOT" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --study-name "$STUDY_NAME" \
  --hu-window -1000 400 \
  --denoise-sigma 0.35 0.70 0.70

# 3. Train and fully evaluate the canonical MedGS model.
"$PYTHON" scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase "$CANONICAL_PHASE" \
  --representation denoised \
  --iterations "$CANONICAL_ITERATIONS" \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100

# 4. Train MedGS4D with phase holdout.
# Train phases: 20, 40, 60, 80
# Validation phases: 10, 30, 50, 70, 90
"$PYTHON" scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --canonical-checkpoint "$CANONICAL_CHECKPOINT" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase "$CANONICAL_PHASE" \
  --target-representation denoised \
  --split-mode phase-holdout \
  --validation-phases 10 30 50 70 90 \
  --iterations "$DYNAMIC_ITERATIONS" \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 250 \
  --validation-samples 20 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda

# 5. Compare representative dynamic checkpoints.
"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --iterations 0 1000 2500 5000 7000 \
  --device cuda

# 6. Diagnose the final learned deformation field.
"$PYTHON" scripts/diagnose_deformation.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$FINAL_DYNAMIC_CHECKPOINT" \
  --device cuda

echo
echo "Completed."
echo "Prepared study: $STUDY_DIR"
echo "Canonical run: $CANONICAL_RUN"
echo "Dynamic run: $RUN_DIR"
echo "Final dynamic checkpoint: $FINAL_DYNAMIC_CHECKPOINT"
echo "Main report: $RUN_DIR/report.pdf"
echo "Checkpoint comparison: $RUN_DIR/evaluation/checkpoint_comparison.pdf"
echo "Deformation diagnostics: $RUN_DIR/evaluation/deformation/iter_$(printf '%06d' "$DYNAMIC_ITERATIONS")/diagnostics.pdf"
