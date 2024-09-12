# PRoC3S
Open-source code-release for paper ["Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction"](https://arxiv.org/abs/2406.05572).

Please reach out to Aidan Curtis (curtisa@csail.mit.edu) and Nishanth Kumar (njk@csail.mit.edu) with any questions!

## Setup
```
conda create -n "proc3s" python=3.10
conda activate proc3s
python -m pip install -e .
```

## Add your OpenAI Key

```
echo "OPENAI_KEY='<YOUR-KEY-HERE>'" > .env
```

## Example commands
The main run file is `eval_policy.py`. Running a particular domain involves simply creating a config file in the `vtamp/config` directory and running `eval_policy.py` using the `--config-dir .` and `--config_name` flags.

Here are a few example commands to give you an idea:

```
# Our approach on a task with goal "draw a rectangle that encloses two obstacles".
python eval_policy.py --config-dir . --config-name=proc3s_draw_star.yaml

# Code as Policies on a RAVENS task with goal "Put three blocks in a line flat on the table"
python eval_policy.py --config-dir=. --config-name=cap_draw_star.yaml

# LLM^3 on a RAVENS task with goal "Put three blocks in a line flat on the table"
python eval_policy.py --config-dir=. --config-name=llm3_draw_star.yaml
```

To turn on caching for llm responses, use the `+policy.use_cache=true` flag. e.g.:

```
python eval_policy.py --config-dir=. --config-name=ours_raven.yaml +policy.use_cache=true
```

Finally, to visualize constraint checking, use the `vis_debug=true` flag. e.g.:
```
python eval_policy.py --config-dir=. --config-name=ours_raven.yaml vis_debug=true ++render=True
```
