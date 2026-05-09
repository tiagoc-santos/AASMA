# AASMA

How to use:

Keep old behavior (default):
python baseline.py

Randomized training partners:
python baseline.py --train_partner_mode random_pool
python baseline.py --train_partner_mode random_pool --deterministic_partner false (Non-deterministic)

Evaluate against a specific partner:
python baseline.py --eval_partner greedy
python baseline.py --eval_partner fetcher
python baseline.py --eval_partner noisy_greedy --eval_partner_epsilon 0.4

Load an existing model and evaluate against non-PPO partner:
python baseline.py --model overcooked_baseline.zip --eval_partner plater