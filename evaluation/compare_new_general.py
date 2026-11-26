"""Experiments: Our training recipe (task 3) at difernetes hiperparametrers (lr, batch size, weight decay, warmup ratio, etc.).

Trained on task 3's split (HF train + multispecies + our data, stratified by species and sigma factor).
Exp1: task 3's test split, plus each model's training-vs-test view. Exp2: test_sigma70, comparison only, with published-2025.
"""
import common

RUNS = ("task3", 
        #"batch_test_32", "batch_test_64", "batch_test_256",
        #"lr_test_1e2", 
        #"lr_test_1e3", "lr_test_1e4", 
        #"lr_test_2e4", "lr_test_4e4", "lr_test_8e4", "lr_test_9e4", 
        #"lr_test_1e5", "lr_test_1e6", "lr_test_1e6_20ep",
        #"wd_test_0001", "wd_test_005", "wd_test_075", "wd_test_075_try2",
        #"wr_test_02", "wr_test_005", "wr_test_015",
        #"do_test_000", "do_test_005", "do_test_015",
        "energy_fusion", "energy_fusion_raw",
        "energy_res", 
        "energy_residual_logits",
        "energy_exp_energy_mlp",
        "energy_exp_energy_mlp_raw",
        "energy_exp_no_energy",
        "energy_w1w2_direct",
        )


def entries(test):
    return [(run, common.read_scores(run, test)) for run in RUNS]


def seeds(test):
    return [(run, m, common.read_scores(m, test)) for run in RUNS for m in common.seed_models(run)]


common.compare(common.ROOT / "results/E1_energy_report", "E: task 3 test split", entries("task3_test"), seeds("task3_test"))
for run in RUNS:
    common.generalisation(common.ROOT / "results/E1_energy_report", run,
                          [("train", common.read_scores(run, "train_task3")), ("test", common.read_scores(run, "task3_test"))])

common.compare(common.ROOT / "results/E2_energy_report_sigma70", "E: sigma70",
               entries("sigma70") + [("published-2025", common.read_scores("published-2025", "sigma70"))], seeds("sigma70"))
