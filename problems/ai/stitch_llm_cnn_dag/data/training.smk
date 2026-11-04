# The training process of the stitched CNN + LM as a rule graph. It is never run by Snakemake: the tensor DAG engine
# (solve.run_engine) parses it with the Circle of Willis problem's parser, builds the rule and job graphs, and runs
# the rules level by level, each rule ONE batched call over its jobs (the models on the stacked model axis).
# Wildcards: {seed} (independent replicates) and {n} (the number of joint image+text training examples).
# Rules without {n} broadcast to every n downstream (the I ⊗ 1 edge); evaluate gathers everything (1ᵀ).

rule train_cnn:
    input: "data/mnist_subset.npz"
    output: "models/cnn_{seed}.pt"

rule train_lm:
    input: "data/sentences.txt"
    output: "models/lm_{seed}.pt"

rule train_monolith:
    input: "data/joint_{n}.npz"
    output: "models/monolith_{seed}_{n}.pt"

rule train_connector:
    input: "models/cnn_{seed}.pt", "models/lm_{seed}.pt", "data/joint_{n}.npz"
    output: "models/connector_{seed}_{n}.pt"

rule global_finetune:
    input: "models/connector_{seed}_{n}.pt", "data/joint_{n}.npz"
    output: "models/stitched_{seed}_{n}.pt"

rule train_router:
    input: "models/connector_{seed}_{n}.pt", "data/joint_{n}.npz"
    output: "models/router_{seed}_{n}.pt"

rule train_router_soft:
    input: "models/connector_{seed}_{n}.pt", "data/joint_{n}.npz"
    output: "models/router_soft_{seed}_{n}.pt"

rule evaluate:
    input: expand("models/cnn_{seed}.pt", seed=SEEDS), expand("models/lm_{seed}.pt", seed=SEEDS), expand("models/connector_{seed}_{n}.pt", seed=SEEDS, n=SIZES), expand("models/stitched_{seed}_{n}.pt", seed=SEEDS, n=SIZES), expand("models/router_{seed}_{n}.pt", seed=SEEDS, n=SIZES), expand("models/router_soft_{seed}_{n}.pt", seed=SEEDS, n=SIZES), expand("models/monolith_{seed}_{n}.pt", seed=SEEDS, n=SIZES)
    output: "results/stages.csv"
