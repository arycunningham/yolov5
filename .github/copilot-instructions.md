## Purpose

This file instructs an AI coding agent how to safely and productively work on the YOLOv5 codebase found at `img_model/yolov5/` in this workspace.
It is intentionally comprehensive: it describes architecture, data flows, developer workflows and exact commands, project-specific conventions, integration points (ROS, loggers), debugging guidance, validation checks, and PR/merge guidelines.

## High-level architecture — components and responsibilities

- CLI entry points (user-facing):
  - `train.py` — training orchestration (dataset loading, augmentation, optimizer, scheduler, loss, checkpointing to `runs/train/`).
  - `detect.py` — inference script for images, videos, streams (camera, RTSP). Handles preprocessing, model inference, NMS, and output formatting.
  - `val.py` — validation/metrics (mAP, precision, recall). Variants exist under `segment/` and `classify/` when applicable.
  - `export.py` — converts PyTorch model checkpoints to ONNX/TensorRT/TFLite/CoreML; includes post-export checks.

- Model definitions:
  - `models/*.yaml` — declarative model specifications (backbone, head, channels, `nc`, anchors). These YAMLs are parsed by the model loader (search for `parse_model`, `Model(` or `yaml.load`).

- Data and hyperparameters:
  - `data/*.yaml` — dataset manifests that include `train`, `val`, `nc`, and `names` fields. Keep these consistent with `models/*.yaml` `nc`.
  - `data/hyps/*.yaml` — hyperparameter files used by training and hyperparameter evolution.

- Utilities & logging:
  - `utils/` — helper code (augmentation, plotting, metrics, dataset utilities). Notable: `utils/flask_rest_api/` and `utils/loggers/{clearml,comet}`.

- Outputs & integrations:
  - `runs/` — training and inference outputs (checkpoints, logs, predictions). Third-party integrations (W&B, ClearML, Comet) are implemented under `utils/loggers/`.
  - `catkin_ws/` — workspace-level ROS packages in this environment; scripts under `catkin_ws/code_dev/` (e.g., `dai_publisher.py`, `dai_publisher_yolov5_runner.py`) expect inference outputs or call model inference directly.

## Data flow (detailed)

1. A `data/*.yaml` manifest points to image/label directories and defines `nc` (num classes) and `names` (class names).
2. `train.py` constructs a `Dataset` and `DataLoader` which apply augmentations from `utils/` or the `datasets/` module.
3. Batches are fed into the model instantiated from `models/*.yaml` via the model loader.
4. Loss computation is handled by `loss.py` (or equivalent); optimizer and scheduler updates are applied by the training loop.
5. Checkpoints (`last.pt`, `best.pt`) are saved in `runs/train/exp/weights/` and logs/metrics are written to `runs/train/exp` and to configured trackers.
6. `detect.py` runs model inference, applies NMS, formats predictions (optional `--save-txt`, `--save-json`), and writes to `runs/detect/exp`.
7. External consumers (ROS nodes, REST API) read inference outputs from the `runs/` layout or via service endpoints in `utils/flask_rest_api/`.

## Key files to inspect before editing (in priority order)

1. `train.py` — training loop and checkpoint logic. Any change here requires smoke training checks.
2. Model loader & `models/*.yaml` — if you change layer definitions, review export and inference code paths.
3. `data/*.yaml` and `data/hyps/*.yaml` — mismatched `nc` is a common source of errors.
4. `utils/loggers/` — update adapters if you add new metrics or change their names/structure.
5. `catkin_ws/code_dev/` and `catkin_ws/src/depthai_publisher/` — verify downstream consumers if changing inference output format or coordinate systems.

## Concrete developer workflows and exact commands

All commands below assume your current directory is `img_model/yolov5/` unless noted.

1) Environment setup

```bash
# using conda
conda create -n yolov5 python=3.10 -y
conda activate yolov5
pip install -r requirements.txt

# If you need a specific CUDA-enabled PyTorch build, install per https://pytorch.org/
```

2) Quick checks / smoke runs

```bash
# Check torch and cuda
python -c "import torch;print('torch', torch.__version__);print('cuda available', torch.cuda.is_available());print('gpus', torch.cuda.device_count())"

# Quick detect smoke (uses yolov5s pretrained weights; will download if missing)
python detect.py --weights yolov5s.pt --source data/images/zidane.jpg --save-txt --save-conf
```

3) Training examples (repro and iteration)

```bash
# Short iteration on COCO128
python train.py --data coco128.yaml --cfg models/yolov5s.yaml --weights '' --epochs 5 --img 640 --batch-size 16

# Full COCO reproduction
python train.py --data coco.yaml --cfg models/yolov5s.yaml --weights '' --epochs 300 --img 640 --batch-size 64

# AutoBatch to let the code pick a safe max batch
python train.py --data coco.yaml --cfg models/yolov5s.yaml --weights '' --epochs 50 --batch-size -1

# Resume training
python train.py --resume runs/train/exp/weights/last.pt

# Multi-GPU DDP (4 GPUs example)
python -m torch.distributed.run --nproc_per_node 4 --master_port 12345 train.py --data coco.yaml --cfg models/yolov5s.yaml --img 640 --batch-size 32 --device 0,1,2,3
```

4) Owner's two-stage fine-tune flow (documented in project notes)

- The file `img_model/YOLOv5_Training_Optimization_Chat_Export.txt` documents a preferred flow used by the owner: Stage 1 (Adam) for ~120 epochs to match evolution metrics (high recall), then Stage 2 (SGD) for ~80 epochs to boost precision. Tuned hyp YAMLs include names like `f2_evolve_38_final.yaml`.

Example:

```bash
# Stage 1 - Adam
python train.py --hyp data/hyps/f2_evolve_38_final.yaml --epochs 120 --optimizer Adam --img 640 --batch-size 32 --weights '' --data data/f2.yaml

# Stage 2 - SGD fine-tune
python train.py --hyp data/hyps/f2_evolve_38_final.yaml --epochs 80 --optimizer SGD --img 640 --batch-size 32 --weights runs/train/exp/weights/best.pt --data data/f2.yaml
```

5) Export and validation

```bash
# Export to ONNX and TensorRT engine
python export.py --weights runs/train/exp/weights/best.pt --include onnx engine --img 640 --device 0 --half

# Validate onnx via onnxruntime
python -c "import onnxruntime as ort; print('onnxruntime device:', ort.get_device())"
```

6) ROS / catkin workflow (workspace-level)

```bash
cd catkin_ws
catkin_make -j4
source devel/setup.bash
# run ROS nodes (roslaunch/rosrun) as required
```

If you modify inference output format or coordinate frames, update the ROS publishers/consumers under `catkin_ws/code_dev/` and `catkin_ws/src/depthai_publisher/`.

## Common pitfalls and detailed mitigations

- CUDA OOM during training
  - Reduce `--batch-size` or `--img` size.
  - Use `--batch-size -1` (AutoBatch) to automatically choose the largest batch that fits GPU memory.
  - Use mixed precision `--half` to reduce memory.
  - For debugging, set `--workers 0` to eliminate additional worker memory footprints.

- Mismatched classes / incorrect `nc`
  - Always verify `data/*.yaml` `nc` equals the model's `nc` (either from `models/*.yaml` or the model instance).
  - Update `names` arrays carefully; downstream loggers or ROS consumers may map indices to class names.

- Failures exporting to ONNX/TensorRT
  - Check for unsupported ops or dynamic control flow in the model. Convert custom ops or implement export-friendly replacements.
  - Try `--simplify` during ONNX export and compare outputs before and after simplification.
  - Ensure TensorRT/ONNXRuntime versions are compatible with the CUDA version on the host.

- Slow or failing DDP
  - Ensure that `torch.distributed.run` arguments match the `--device` selection and that NCCL environment variables are set correctly in CI.

## Testing and validation checklist for agent-made edits

Before opening a PR, an agent should run (or provide reproducible commands for) the following checks:

1. Static checks
   - Run any configured linters or type-checkers (search repo for `.flake8`, `mypy.ini`, `pyproject.toml`).

2. Unit/smoke tests
   - Add or run a smoke test that:
     - Imports the modified code.
     - Calls `detect.py` on `data/images/zidane.jpg` and asserts `runs/detect/exp` contains output files.
     - Runs `train.py --data coco128.yaml --epochs 1` to ensure one-epoch sanity training completes and `runs/train/exp` has a checkpoint.

3. Functional checks
   - For model changes, run `export.py` to produce ONNX and perform a lightweight inference with ONNXRuntime.

4. Integration checks (if applicable)
   - If inference I/O changed, run the corresponding ROS node(s) locally and validate published topics or output files.

If any check fails, include a minimal reproduction script in the PR or comment describing how to reproduce the failure locally.

## Expanded agent "contract" (inputs/outputs/guardrails)

- Inputs: code changes, new files (e.g., `models/*.yaml`, new utils), or updated hyperparameters.
- Allowed side-effects: adding tests, docs, or lightweight utility scripts in `tools/`.
- Required outputs for PR creation:
  - All smoke tests above pass locally.
  - No silent behavioural changes in CLI signatures without documentation updates.
  - If model architecture changes: provide a note describing export impact and required downstream updates (ROS, REST, consumers).
- Guardrails:
  - Do not commit large datasets or binary checkpoints.
  - Avoid changing the `runs/` output layout (filenames like `best.pt` and `last.pt`) unless all consumers are updated.

## PR checklist for low-risk merges

1. Update README or this instruction file if CLI flags or default behavior changed.
2. Add new Python deps to `requirements.txt` and document rationale.
3. Add or update smoke tests under `tests/` and include commands to reproduce locally.
4. If model changes, include an `export.py` smoke export and note downstream consumers.
5. Ensure `img_model/YOLOv5_Training_Optimization_Chat_Export.txt` is not modified unless you own the experiments — instead reference it in the PR description.

## Useful quick commands (reference)

```bash
# GPU check
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"

# Detect smoke
python detect.py --weights yolov5s.pt --source data/images/zidane.jpg --save-txt --project runs/detect/debug

# Train smoke (1 epoch)
python train.py --data coco128.yaml --cfg models/yolov5s.yaml --weights '' --epochs 1 --img 320 --batch-size 8

# Export
python export.py --weights runs/train/exp/weights/best.pt --include onnx --img 640

# Rebuild catkin after ROS changes
cd catkin_ws && catkin_make && source devel/setup.bash
```

## Repo-specific references and owner notes

- `img_model/YOLOv5_Training_Optimization_Chat_Export.txt` — contains owner notes on hyperparameter evolution, tuned YAML filenames, and the Adam→SGD handoff. Use the listed YAML names (e.g., `f2_evolve_38_final.yaml`) to reproduce experiments.
- `utils/loggers/` — adapters and examples for experiment tracking systems; follow their patterns when adding telemetry.
- `catkin_ws/code_dev/` — example integration code showing how inference outputs are consumed by ROS/DepthAI.

## Offer to expand

I can expand this file with any of the following on request:

- A smoke-test harness under `tests/smoke/` that runs detect + one-epoch train + export and reports a pass/fail.
- A GitHub Action workflow that runs the CUDA/PyTorch smoke checks on PRs to `master`.
- A short migration guide for converting `models/*.yaml` changes into export-friendly ONNX/TensorRT models (with example diffs).

If you want this content moved to the repository root `.github/copilot-instructions.md` (global guidance) or split across multiple project-specific files (one for `yolov5/`, one for `catkin_ws/`), say which layout you prefer and I'll apply the change.
## Purpose

Short, targeted instructions to help an AI coding agent be productive in this YOLOv5 codebase (located at the repo root). Focus: big-picture architecture, common workflows, patterns, and exact file locations to inspect before editing.

## Big picture (what to know first)

- This repo is a standard Ultralytics YOLOv5 codebase customized inside `img_model/yolov5/` in this workspace. Primary entry points: `train.py`, `detect.py`, `val.py`, `export.py` and the `segment/` and `classify/` subfolders. Models live in `models/`, dataset manifests and hyps live in `data/` and `data/hyps/`.
- There is also ROS integration in the workspace: `catkin_ws/` contains multiple ROS packages (not inside this fork) and `catkin_ws/code_dev/` contains helper scripts such as `dai_publisher.py` that interact with DepthAI nodes. Changes to model I/O or serialization may affect those ROS scripts.
- Experiment outputs and artifacts are saved to `runs/` (e.g. `runs/train`, `runs/detect`). Logging integrations live under `utils/loggers/` (ClearML, Comet, etc.).

## Critical files & directories (inspect these first)

- `train.py`, `detect.py`, `val.py`, `export.py` — main CLI scripts. Flags are consistent across them (`--weights`, `--data`, `--img`, `--batch-size`, `--device`).
- `models/` — model YAML definitions (backbone / head). Changing architecture there affects training and export.
- `data/` — dataset YAMLs and hyperparameter files `data/hyps/*.yaml`.
- `utils/` — common helpers; `utils/flask_rest_api/` is a lightweight service integration; `utils/loggers/{clearml,comet}/` implement experiment trackers.
- `img_model/YOLOv5_Training_Optimization_Chat_Export.txt` — local notes / hyperparameter evolution artifacts (useful when reproducing tuned runs).
- `catkin_ws/code_dev/` and `catkin_ws/src/depthai_publisher/` — ROS publisher nodes that consume model outputs or camera streams. When changing inference API or output format, update consumers here.

## Developer workflows and commands (exact examples)

1) Create Python environment and install dependencies (run from `img_model/yolov5`):

```bash
# create/activate virtualenv or conda env then:
pip install -r requirements.txt
```

2) Quick smoke to check CUDA and PyTorch availability:

```bash
python -c "import torch;print('cuda', torch.cuda.is_available());print(torch.__version__)"
```

3) Train (examples from the repo README):

```bash
# Single-GPU
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml --batch-size 64

# AutoBatch (auto pick max batch that fits):
python train.py --data coco.yaml --epochs 50 --weights '' --cfg yolov5s.yaml --batch-size -1

# Distributed Data Parallel (multi-GPU)
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco.yaml --img 640 --device 0,1,2,3
```

4) Inference / detect examples:

```bash
python detect.py --weights yolov5s.pt --source 0          # webcam
python detect.py --weights yolov5s.pt --source path/to/imgs/  # images
```

5) ROS / system integration notes:

- Build workspace when editing ROS nodes: run `catkin_make` from `catkin_ws/` and then `source devel/setup.bash` before running ROS nodes.
- Many ROS helper scripts in `catkin_ws/code_dev/` expect model files or inference outputs in standard YOLO `runs/` layout; preserve output JSON/CSV formats when refactoring inference code.

## Project-specific conventions and patterns

- CLI parity: scripts expose similar flags (`--weights`, `--data`, `--img`, `--batch-size`, `--device`) and use YAML data files under `data/` for datasets. Follow this convention when adding new scripts.
- Hyperparameters live in `data/hyps/`. Hyperparameter evolution code expects these YAMLs.
- Logging: experiments are routed to `runs/` and exported via logger adapters in `utils/loggers/`. Prefer using existing logger hooks when adding metrics.
- Model definitions are pure YAML in `models/` — changing layer names or expected anchor formats must be mirrored in export and inference.

## Integration points & external dependencies

- PyTorch + CUDA (GPU training). Confirm `torch.cuda.is_available()` before long runs.
- Optional trackers: Weights & Biases, Comet, ClearML — configured under `utils/loggers/`.
- ROS (catkin) packages live under `catkin_ws/`. Changes to inference signatures may require updates to DepthAI publishers / subscribers.

## Small agent "contract" (what edits should do / validate)

- Inputs: edits to scripts or models. Outputs: no new runtime errors, preserved CLI behavior, and tests/few-shot smoke runs succeed.
- Success criteria: `python -c "import torch; print(torch.cuda.is_available())"` runs; `python detect.py --weights yolov5s.pt --source img.jpg` executes without crashing and writes to `runs/detect/`.
- Error modes to watch: CUDA OOM (reduce `--batch-size` or `--img`), missing YAML keys in `data/` or `models/`, mismatched tensor shapes after model changes.

## Quick debugging tips (project-specific)

- OOM on training: reduce `--batch-size`, use `--img` smaller size, use `--batch-size -1` (AutoBatch), or enable `--half` (fp16) for mixed precision.
- Mismatched anchors / classes: check dataset YAML (`data/*.yaml`) and `models/*.yaml` class count. Many downstream scripts assume consistent `nc` (num classes).
- Export issues (ONNX/TensorRT): check that `models/` YAMLs use supported layers and run `export.py --weights runs/train/exp/weights/best.pt --include onnx` to reproduce.

## Where to look for more context

- `img_model/YOLOv5_Training_Optimization_Chat_Export.txt` — local notes about hyperparameter evolution and tuned YAMLs used by the owner.
- `img_model/yolov5/README.md` — canonical usage and examples (training, export, integrations).
- `catkin_ws/code_dev/` — glue scripts that show how model outputs are consumed by ROS nodes.

---

If anything in these sections is unclear or you want more detail for a particular change (for example, exact unit tests to run, or to scaffold a small smoke test harness), tell me which area and I'll iterate.
