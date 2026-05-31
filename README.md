# AASMA

Train and evaluate an ego agent for ad-hoc teamwork in Overcooked. The main entry point is `src/training.py`, which can both train a new model and evaluate/render gameplay.

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the updated library files into the virtual environment (this is needed since the original overcooked-ai library does not offer support for more than 2 players):

```bash
PYTHON=python3
VENV_SITE=$($PYTHON -c "import site; print(site.getsitepackages()[0])")
rsync -a overcooked_ai_py_new/ "$VENV_SITE/overcooked_ai_py/" || cp -R overcooked_ai_py_new/. "$VENV_SITE/overcooked_ai_py/"
```

Run with defaults (trains a model, evaluates it, saves results and a GIF). Commands below assume you run from `src/` so relative paths (for layouts, models, heatmaps, GIFs, CSVs) resolve inside the repo:

```bash
cd src
python training.py
```

## What happens when you run training.py

- If `--model` is not provided, a new model is trained and saved under `models/<architecture>_<train_partner_mode>_seed<seed>/` as `models/<architecture>_<train_partner_mode>_seed<seed>/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>.zip`.
- VecNormalize statistics are saved alongside the model as `models/<architecture>_<train_partner_mode>_seed<seed>/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>_vecnormalize.pkl`.
- Evaluation always runs after training/loading.
- A gameplay GIF is saved to `gameplay_gifs/<train_mode_label>_seed<seed>/<architecture>_<train_mode_label>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.gif`.
- A heatmap PDF is saved to `heatmaps/<train_mode_label>_seed<seed>/<architecture>_<train_mode_label>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.pdf`.
- Evaluation summaries are saved (or updated) in `evaluation_results.csv`.
- When `--model` is used, `train_mode_label` is derived from the filename stem. If it matches `<architecture>_<label>_<timesteps>_seed<seed>.zip`, the label becomes `<label>` (for models saved by this script, this is just `<train_partner_mode>`). Otherwise, it falls back to `loaded_model`.
- When `--train_partner_mode` is `adhoc_curriculum`, intermediate checkpoints are also saved in the model directory: `adhoc_low_checkpoint.zip`, `adhoc_mid_checkpoint.zip`, and `adhoc_final_checkpoint.zip`, plus matching `_vecnormalize.pkl` files.

## Common usage patterns

Train with randomized partner teams:

```bash
python training.py --train_partner_mode random_pool
```

Train with an CNN policy instead of RNN:

```bash
python training.py --architecture cnn
```

Evaluate against a specific partner team:

```bash
python training.py --eval_partner greedy
python training.py --eval_partner specialists
python training.py --eval_partner noisy_greedy --eval_partner_epsilon 0.4
```

Load an existing model and evaluate it:

```bash
python training.py --model ../models/cnn_self_play_seed42/cnn_self_play_2000000_seed42.zip --eval_partner greedy
```

Use a custom layout (local file or built-in layout name):

```bash
python training.py --layout_name three_chefs
```

Run the three-stage ad-hoc curriculum:

```bash
python training.py --train_partner_mode adhoc_curriculum
```

Evaluate with the validation suite:

```bash
python training.py --eval_suite common
```

Evaluate with the test partner suite:

```bash
python training.py --eval_suite test
```

Run passive-ego baseline evaluation against test teams:

```bash
python passive_baseline.py \
	--model ../models/rnn_adhoc_curriculum_seed42/rnn_adhoc_curriculum_3000000_seed42.zip \
	--model_label self_play \
	--model_seed 42 \
	--suite unseen_final
```

## Flags

All flags are optional; defaults are shown below.

- `--timesteps` (int, default: 2000000)
	- Total timesteps to train when creating a new model.
- `--model` (str, default: None)
	- Path to a pre-trained model zip. If provided, training is skipped and the model is evaluated.
	- Note: when `--model` is used, `architecture` is inferred from the first filename segment only if it is `cnn`, `mlp`, or `rnn`. The `train_mode_label` used in outputs is derived from filenames that match `<architecture>_<label>_<timesteps>_seed<seed>.zip`; otherwise it is `loaded_model`.
- `--layout_name` (str, default: three_chefs)
	- Overcooked layout name. Can refer to a built-in layout or a local file in `layouts/` (from `src/`, this resolves to `../layouts/<name>.layout`).
- `--train_partner_mode` (str, default: self_play)
	- Partner schedule during training. Choices: `self_play`, `random_pool` or `adhoc_curriculum`.
- `--train_noisy_epsilon` (float, default: 0.25)
	- Epsilon used for the `noisy_greedy` partners in the training pool.
- `--eval_partner` (str, default: ppo)
	- Partner team used during evaluation and GIF rendering. Choices: `ppo`, `random`, `stationary`, `greedy`, `specialists`, `noisy_greedy`, `heldout_noisy_greedy`, `heldout_greedy_noisy`, `alternating_cookers`, `prepositioning_servers`, `role_switchers`, `yielding_generalists`.
- `--eval_partner_epsilon` (float, default: 0.25)
	- Epsilon used when `--eval_partner noisy_greedy`.
- `--eval_suite` (str, default: single)
	- Evaluation suite. Choices: `single` (use `--eval_partner`), `common` (greedy, specialists, heldout_noisy_greedy, heldout_greedy_noisy), or `test` (alternating_cookers, prepositioning_servers, role_switchers, yielding_generalists).
	- Note: the `test` suite partners expect a single-pot layout and will raise an error otherwise.
- `--deterministic_partner` (str, default: false)
	- Whether partner policies act deterministically during evaluation/rendering. Choices: `true`, `false`.
- `--eval_episodes` (int, default: 100)
	- Number of evaluation episodes.
- `--results_csv` (str, default: evaluation_results.csv)
	- Output CSV for evaluation summaries.
- `--num_cpu` (int, default: 4)
	- Number of CPU cores used for training environments.
- `--architecture` (str, default: rnn)
	- Policy architecture. Choices: `mlp`, `cnn`, `rnn`.
- `--deterministic_ego` (str, default: true)
	- Whether the ego policy acts deterministically during evaluation/rendering. Choices: `true`, `false`.
- `--seed` (int, default: 42)
	- Random seed used for training and evaluation.

Additional flags for passive baseline evaluation (`src/passive_baseline.py`):

- `--model` (str, required)
	- Path to a pre-trained model zip.
- `--model_label` (str, required)
	- Label written to the output CSV (use the training mode or model family name).
- `--model_seed` (int, required)
	- Seed used for the trained model; inferred from the filename when possible.
- `--layout_name` (str, default: three_chefs)
	- Layout used for evaluation.
- `--architecture` (str, default: rnn)
	- Policy architecture; inferred from the model filename when it starts with `cnn`, `mlp`, or `rnn`.
- `--eval_episodes` (int, default: 100)
	- Number of evaluation episodes per team.
- `--suite` (str, default: both)
	- Team suite to evaluate. Choices: `compatibility`, `contribution`, `unseen_final`, `both`.
- `--deterministic_ego` (str, default: true)
	- Whether the trained ego acts deterministically.
- `--deterministic_partner` (str, default: false)
	- Whether partner agents act deterministically.
- `--output_csv` (str, default: ../passive_baseline_results.csv)
	- Output CSV file for passive baseline results.

## Outputs

- Trained models: `models/<architecture>_<train_partner_mode>_seed<seed>/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>.zip`
- VecNormalize stats: `models/<architecture>_<train_partner_mode>_seed<seed>/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>_vecnormalize.pkl`
- Gameplay GIFs: `gameplay_gifs/<train_mode_label>_seed<seed>/<architecture>_<train_mode_label>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.gif`
- Heatmaps: `heatmaps/<train_mode_label>_seed<seed>/<architecture>_<train_mode_label>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.pdf`
- Ad-hoc curriculum checkpoints: `models/<architecture>_adhoc_curriculum_seed<seed>/adhoc_<low|mid|final>_checkpoint.zip` and matching `_vecnormalize.pkl` files
- Evaluation results: `evaluation_results.csv`
- Passive baseline results: `passive_baseline_results.csv`