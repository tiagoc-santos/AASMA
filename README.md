# AASMA

Train and evaluate an ego agent for ad-hoc teamwork in Overcooked. The main entry point is `src/training.py`, which can both train a new model and evaluate/render gameplay.

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the updated library files into the virtual environment (this is needed since the original overcooked-ai library does not offer support for more than 2 player):

```bash
PYTHON=python3
VENV_SITE=$($PYTHON -c "import site; print(site.getsitepackages()[0])")
rsync -a overcooked_ai_py_new/ "$VENV_SITE/overcooked_ai_py/" || cp -R overcooked_ai_py_new/. "$VENV_SITE/overcooked_ai_py/"
```

Run with defaults (trains a model, evaluates it, saves results and a GIF):

```bash
python src/training.py
```

## What happens when you run src/training.py

- If `--model` is not provided, a new model is trained and saved to `models/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>.zip`.
- VecNormalize statistics are saved alongside the model as `models/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>_vecnormalize.pkl`.
- Evaluation always runs after training/loading.
- A gameplay GIF is saved to `gameplay_gifs/<architecture>_<train_partner_mode>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.gif`.
- A heatmap PDF is saved to `heatmaps/<architecture>_<train_partner_mode>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.pdf`.
- Evaluation summaries are saved (or updated) in `evaluation_results.csv`.

## Common usage patterns

Train with randomized partner teams:

```bash
python training.py --train_partner_mode random_pool
```

Train with a CNN policy instead of an MLP:

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
python src/training.py --model models/cnn_curriculum_2000000_seed42.zip --eval_partner greedy
```

Use a custom layout (local file or built-in layout name):

```bash
python src/training.py --layout_name three_chefs
```

Run the three-stage ad-hoc curriculum:

```bash
python src/training.py --train_partner_mode adhoc_curriculum
```

Evaluate with the fixed external suite:

```bash
python src/training.py --eval_suite common
```

## Flags

All flags are optional; defaults are shown below.

- `--timesteps` (int, default: 2000000)
	- Total timesteps to train when creating a new model.
- `--model` (str, default: None)
	- Path to a pre-trained model zip. If provided, training is skipped and the model is evaluated.
	- Note: when `--model` is used, `architecture` and `train_partner_mode` are inferred from the filename stem in the format `<architecture>_<train_partner_mode>_<timesteps>.zip` and used for GIF naming and CSV rows.
- `--layout_name` (str, default: three_chefs)
	- Overcooked layout name. Can refer to a built-in layout or a local file in `layouts/`.
- `--train_partner_mode` (str, default: curriculum)
	- Partner schedule during training. Choices: `curriculum`, `random_pool` or `adhoc_curriculum`.
- `--train_noisy_epsilon` (float, default: 0.25)
	- Epsilon used for the `noisy_greedy` partners in the training pool.
- `--eval_partner` (str, default: ppo)
	- Partner team used during evaluation and GIF rendering. Choices: `ppo`, `random`, `stationary`, `greedy`, `specialists`, `noisy_greedy`, `heldout_noisy_greedy`, `heldout_greedy_noisy`.
- `--eval_partner_epsilon` (float, default: 0.25)
	- Epsilon used when `--eval_partner noisy_greedy`.
- `--eval_suite` (str, default: single)
	- Evaluation suite. Choices: `single` (use `--eval_partner`) or `common` (fixed suite).
- `--deterministic_partner` (str, default: false)
	- Whether partner policies act deterministically during evaluation/rendering. Choices: `true`, `false`.
- `--eval_episodes` (int, default: 100)
	- Number of evaluation episodes.
- `--results_csv` (str, default: evaluation_results.csv)
	- Output CSV for evaluation summaries.
- `--num_cpu` (int, default: 4)
	- Number of CPU cores used for training environments.
- `--architecture` (str, default: cnn)
	- Policy architecture. Choices: `mlp`, `cnn`.
- `--deterministic_ego` (str, default: true)
	- Whether the ego policy acts deterministically during evaluation/rendering. Choices: `true`, `false`.
- `--seed` (int, default: 42)
	- Random seed used for training and evaluation.

## Outputs

- Trained models: `models/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>.zip`
- VecNormalize stats: `models/<architecture>_<train_partner_mode>_<timesteps>_seed<seed>_vecnormalize.pkl`
- Gameplay GIFs: `gameplay_gifs/<architecture>_<train_partner_mode>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.gif`
- Heatmaps: `heatmaps/<architecture>_<train_partner_mode>_<eval_partner>_seed<seed>_<ego_det|ego_stoch>_<partner_det|partner_stoch>.pdf`
- Evaluation results: `evaluation_results.csv`