# The training process of the chest X-ray report toy as a rule graph. It is never run by Snakemake: the tensor DAG
# engine (solve.run_engine) parses it with the Circle of Willis problem's parser, builds the rule and job graphs, and
# runs the rules level by level, each rule ONE batched call over its jobs (the seeds on the stacked model axis).
# Wildcard: {seed} (independent replicates). evaluate gathers everything (1ᵀ).

rule train_cnn:
    input: "data/nih_cxr_subset.npz"
    output: "models/cnn_{seed}.pt"

rule train_lm:
    input: "data/template_reports.txt"
    output: "models/lm_{seed}.pt"

rule train_monolith:
    input: "data/nih_cxr_subset.npz"
    output: "models/monolith_{seed}.pt"

rule train_connector:
    input: "models/cnn_{seed}.pt", "models/lm_{seed}.pt", "data/nih_cxr_subset.npz"
    output: "models/connector_{seed}.pt"

rule global_finetune:
    input: "models/connector_{seed}.pt", "data/nih_cxr_subset.npz"
    output: "models/stitched_{seed}.pt"

rule train_router:
    input: "models/connector_{seed}.pt", "data/nih_cxr_subset.npz"
    output: "models/router_{seed}.pt"

rule caption_finetune:
    input: "models/stitched_{seed}.pt", "data/pmc_cxr_subset.npz", "data/pmc_captions.csv"
    output: "models/captioner_{seed}.pt"

rule evaluate:
    input: expand("models/cnn_{seed}.pt", seed=SEEDS), expand("models/lm_{seed}.pt", seed=SEEDS), expand("models/monolith_{seed}.pt", seed=SEEDS), expand("models/connector_{seed}.pt", seed=SEEDS), expand("models/stitched_{seed}.pt", seed=SEEDS), expand("models/router_{seed}.pt", seed=SEEDS), expand("models/captioner_{seed}.pt", seed=SEEDS)
    output: "results/stages.csv"
