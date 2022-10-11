import os
import pandas as pd
import numpy as np
    
    
def calc_overall_results(): # debug code
    exp = 'results_iclr/MINI_DOMAIN_NET/DDC/seed=2/experiments_logs/multirun/da'
    # for exp in experiments:
    results = pd.DataFrame(
            columns=["scenario", "lambda", "src_acc", "src_f1", "trg_acc", "trg_f1", "src_risk", "trg_risk", "dev_risk"])

    single_exp = os.listdir(exp)
    single_exp = [i for i in single_exp if "_tgt-" in i and 'none' not in i]
    single_exp.sort()

    src_ids = [single_exp[i].split("-")[2] for i in range(len(single_exp))]
    scenarios_ids = np.unique(["_".join(i.split("-")[2:4]) for i in single_exp])
    num_runs = len(src_ids) // len(scenarios_ids)

    for scenario in single_exp:
        scenario_dir = os.path.join(exp, scenario)
        scores = pd.read_excel(os.path.join(scenario_dir, 'scores.xlsx'))
        results = results.append(scores)
        name = '_'.join(scenario.split('-')[:-1])
        idx = 0
        for s in scenarios_ids:
            if s in name:
                break
            idx += 1
        results.iloc[len(results) - 1, 0] = scenarios_ids[idx]
        results.iloc[len(results) - 1, 1] = scenario.split('-')[-1]

    avg_results = results.groupby(np.arange(len(results)) // num_runs).mean()
    std_results = results.groupby(np.arange(len(results)) // num_runs).std()

    avg_results.insert(0, "scenario", list(scenarios_ids), True)
    avg_results.loc[len(avg_results)] = avg_results.mean()
    avg_results = avg_results.fillna('Mean')
    std_results.insert(0, "scenario", list(scenarios_ids), True)

    report_save_path_all = os.path.join(exp, f"all_results.xlsx")
    report_save_path_avg = os.path.join(exp, f"average_results.xlsx")
    report_save_path_std = os.path.join(exp, f"std_results.xlsx")

    results.to_excel(report_save_path_all)
    avg_results.to_excel(report_save_path_avg)
    std_results.to_excel(report_save_path_std)


if __name__ == '__main__':
    calc_overall_results()
