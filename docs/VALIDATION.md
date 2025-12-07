# Validating Custom Changes

Use the steps below to verify that new planner logic, LLM settings, or skills you add are working as expected. The checklist starts from a clean repo state and finishes with artifact inspection so you can confirm your innovation behaves differently from the baseline.

## 1. Prepare the environment
- Activate the project environment (matches the versions in `INSTALLATION.md`):
  ```bash
  conda activate habitat-llm
  ```
- Clear any GPU device masking you added during debugging so both agents can see the same devices:
  ```bash
  unset CUDA_VISIBLE_DEVICES
  export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/egl_nvidia/10_nvidia.json"
  ```

## 2. Set dataset, models, and episode selection
- Point the dataset path to the split you want to test. For quick validation, the mini validation set keeps runtime short:
  ```bash
  DATA_PATH="/space0/hecb/partnr-planner/data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
  ```
- Select the episodes that should expose your change. Example: first 10 episodes stored in an env var for reuse:
  ```bash
  EPISODES="[0,1,2,3,4,5,6,7,8,9]"
  ```
- Choose your customized LLM weights (local or HF ID). Replace these paths with your model when validating LLM innovations:
  ```bash
  MODEL_PATH="/space0/hecb/partnr-planner/models/Llama-3.1-8B-Instruct"
  ```

## 3. Run the decentralized multi-agent demo with your changes
Execute the planner demo while pointing to your modified code and model. This command saves both videos and trajectory JSON so you can inspect behavior differences:

```bash
HYDRA_FULL_ERROR=1 \
python -m habitat_llm.examples.planner_demo \
  --config-name baselines/decentralized_zero_shot_react_summary.yaml \
  habitat.dataset.data_path="$DATA_PATH" \
  evaluation.save_video=True \
  evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
  evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
  evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine="$MODEL_PATH" \
  evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine="$MODEL_PATH" \
  trajectory.save=True \
  trajectory.save_path="data/coalign_demo_10eps" \
  trajectory.save_options="[rgb,depth,pose]" \
  +episode_indices="$EPISODES"
```

### Where to plug in your innovation
- **Planner or decision logic changes:** ensure the CLI references your modified config keys (e.g., `plan_config.decision_hooks.*` or new Hydra overrides). If you added new flags, append them to the command and rerun.
- **Custom LLM or prompt changes:** point `generation_params.engine` to your model or update prompt templates in your code, then rerun to observe changed trajectories.
- **Skill or action-level changes:** if you introduced NN skills, add the corresponding agent config and checkpoints to the command (see `README` quickstart for nn agents), then validate with the same steps above.

## 4. Inspect results
- Videos and trajectories are written to `data/coalign_demo_10eps` by default. Compare them against a baseline run to confirm your modification’s effect.
- For quick quantitative checks or debugging, read the trajectory JSON:
  ```bash
  python scripts/read_results.py data/coalign_demo_10eps/partnr_episodes
  ```
- If your change targets belief or concept handling, use the visualization helper:
  ```bash
  python scripts/visualize_belief_divergence.py data/coalign_demo_10eps/partnr_episodes
  ```

## 5. Speed tips
- Start with the mini validation split and a small episode subset (e.g., 0–2) to iterate quickly.
- Set `evaluation.save_video=False` during rapid debugging to skip video rendering overhead.

Follow this sequence each time you modify the code to confirm the new behavior and capture artifacts that reflect your innovation.
