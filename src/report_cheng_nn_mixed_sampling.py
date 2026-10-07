"""Read completed mixed-sampling runs; make a compact comparison notebook.

No training, test loading, RL evaluation, model publication or significance
testing is performed. Error bars/SD refer to training seeds, not case CIs.
"""
from pathlib import Path
import argparse
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_cheng_nn_mixed_sampling import (
    DEFAULT_OUTPUT, SCENARIOS, read_json, validate_manifest, validate_run,
)
from cheng_nn_simulation import COEFFICIENT_NAMES
from train_cheng_nn import atomic_csv, atomic_json, file_sha256


def build_report(output):
    output = Path(output).resolve()
    manifest = read_json(output/'experiment.json')
    if manifest['status'] != 'completed' or len(manifest['runs']) != 6:
        raise ValueError('Wait until all six models have completed.')
    validate_manifest(manifest)
    for entry in manifest['runs']:
        validate_run(entry, manifest)
    summary = pd.read_csv(output/'comparison_summary.csv').set_index('dimension')
    if not (summary.completed_seeds == 3).all():
        raise ValueError('Need three paired seeds for each input.')
    table = pd.DataFrame({
        'coefficient': COEFFICIENT_NAMES,
        'RMSE_17D': [summary.loc[17, f'{c}_mean'] for c in COEFFICIENT_NAMES],
        'RMSE_21D': [summary.loc[21, f'{c}_mean'] for c in COEFFICIENT_NAMES],
    })
    table['improvement_21D_percent'] = 100*(1-table.RMSE_21D/table.RMSE_17D)
    atomic_csv(output/'coefficient_comparison.csv', table)
    groups = pd.read_csv(output/'scenario_summary.csv')
    group_table = groups.pivot(index=['scenario','coefficient'], columns='dimension', values='RMSE_mean')
    group_table.columns = ['RMSE_17D','RMSE_21D']
    group_table['improvement_21D_percent'] = 100*(1-group_table.RMSE_21D/group_table.RMSE_17D)
    atomic_csv(output/'scenario_comparison.csv', group_table.reset_index())
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    seeds = sorted({r['seed'] for r in manifest['runs']})
    colors = ['#2374ab','#e48c24','#32936f']
    for ax, dim in zip(axes, [17,21]):
        for entry in manifest['runs']:
            if entry['dimension'] != dim:
                continue
            history = pd.read_csv(Path(entry['path'])/'training_history.csv')
            color = colors[seeds.index(entry['seed'])]
            ax.plot(history.epoch, history.validation_RMSE_eval, color=color, label=str(entry['seed']))
            ax.plot(history.epoch, history.train_RMSE_eval, color=color, ls='--', alpha=.7)
        ax.set(title=f'{dim}D / four equally weighted scenarios', xlabel='Epoch')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Coefficient RMSE (train-only target z units)')
    axes[0].legend(title='Training seed', fontsize=8)
    fig.suptitle('Solid = validation; dashed = train; same end-of-epoch weights; no test')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png',dpi=160)
    plt.close(fig)

    def md(text):
        return dict(cell_type='markdown',metadata={},source=text.splitlines(True))
    def code(text, count, rendered):
        return dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=count,
                    outputs=[dict(output_type='display_data',metadata={},data=rendered)])
    def table_cell(filename, frame, count):
        return code(f"display(pd.read_csv(ROOT / {filename!r}).round(5))", count,
                    {'text/plain':frame.round(5).to_string(index=False),
                     'text/html':frame.round(5).to_html(index=False)})
    cells = [md('# 新混合抽樣：17D／21D 小模型比較\n\n'
        '同一批 100,000 組，1976–2025 共50年；train/validation/test=80k/10k/10k。'
        '四種情境 M0、M_mu、M_sigma、M_mu_sigma 各25%；比例是實驗設計，不代表全球真實占比。'
        '情境標籤不餵入 NN。test 已生成但未讀取／評估。\n\n'
        '固定 128–128–64、Adam lr=1e-3、L2=1e-4、Dropout=0、batch128；'
        '最多300 epochs、patience20、ReduceLROnPlateau；三paired seeds。'
        '以 validation 係數 MSE 選 checkpoint；不評估RL，不發布正式模型。'),
        dict(cell_type='code',metadata={},execution_count=1,outputs=[],source=[
            'from pathlib import Path\n','import pandas as pd\n',
            'from IPython.display import display, Image\n',f'ROOT = Path({str(output)!r})\n']),
        md('## 五係數 validation RMSE\n\n各欄為三個seed的RMSE平均。'
           '改善率=100×(1−RMSE_21D/RMSE_17D)，正值代表21D較小；未做顯著性檢定。'
           '不能與舊抽樣資料的RMSE直接相減，因驗證分布不同。'),
        table_cell('coefficient_comparison.csv',table,2),
        md('## 分情境比較\n\n每種情境是同一批2,500組validation；三個seed不是7,500個獨立案例。'
           '特別注意兩斜率都不為零時是否惡化，避免只靠零情境拉低整體RMSE。'
           '精確零真值案例的RMSE衡量誤估程度，不等於假設檢定的偽陽性率。'),
        table_cell('scenario_comparison.csv',group_table.reset_index(),3),
        md('## 過擬合與學習曲線\n\n五個係數單位不同，不直接相加原尺度RMSE。'
           '合併RMSE使用同一份train-only target scaler；gap=100×(validation/train−1)。'),
        table_cell('rmse_per_seed.csv',pd.read_csv(output/'rmse_per_seed.csv'),4),
        code("display(Image(filename=str(ROOT/'learning_curves.png')))",5,
             {'image/png':base64.b64encode((output/'learning_curves.png').read_bytes()).decode(),
              'text/plain':'Matched train/validation learning curves'}),
        md('## GEV 支持範圍\n\n區分參數無效與觀測值不在預測support內；所有finite係數預測仍納入RMSE，'
           '不因support違反刪除案例。xi不必大於零。'),
        table_cell('gev_validity_summary.csv',pd.read_csv(output/'gev_validity_summary.csv'),6)]
    atomic_json(output/'cheng_NN_mixed_sampling_comparison.ipynb',dict(cells=cells,
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        nbformat=4,nbformat_minor=4))
    atomic_json(output/'report_metadata.json',dict(status='completed',test_used=False,
        n_models=6,source_sha256=file_sha256(Path(__file__)),
        experiment_sha256=file_sha256(output/'experiment.json')))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    build_report(parser.parse_args().output)
