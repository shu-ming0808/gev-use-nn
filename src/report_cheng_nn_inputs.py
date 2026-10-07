"""Figures, grouped tables and matched CPU timing for the input experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from compare_cheng_nn_inputs import DEFAULT_OUT, read_json, make_features, load_data
from train_cheng_nn import atomic_csv, atomic_json, file_sha256
from cheng_nn_simulation import COEFFICIENT_NAMES

LABELS=[r'$\mu_0$',r'$\beta_\mu$',r'$\eta_0$',r'$\beta_\sigma$',r'$\xi_0$']
COLORS=['#377eb8','#e68613','#2c9956']


def benchmark(output, manifest, ns, data):
    """Same machine/thread count/batch for all models, no model loading time.

    End-to-end timing starts from in-memory annual maxima, recomputes sample
    median/IQR + features, and ends with original-scale coefficients. It
    excludes disk I/O and includes data copying and target inverse transforms.
    """
    torch.set_num_threads(4)
    rows=[]
    values=np.asarray(data['validation']['annual_maxima'])
    def preprocess(dim):
        q=np.quantile(values,[.25,.5,.75],axis=1)
        scale=np.column_stack([q[1],q[2]-q[0]])
        return make_features(values,scale,dim),scale
    for entry in manifest['runs']:
        dim,seed,path=entry['dimension'],entry['seed'],Path(entry['path'])
        if file_sha256(path/'best.pt')!=entry['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs from completed experiment.')
        ns['INPUT_FEATURE_NAMES']=tuple(f'input_{i}' for i in range(dim))
        model,scaler,_=ns['load_cheng_checkpoint'](path/'best.pt','cpu')
        x,scale=preprocess(dim)
        saved=np.load(path/'validation/predictions_standardized.npy')
        cpu=ns['predict_standardized_coefficients'](model,x,scaler,'cpu',batch_size=2048)
        np.testing.assert_allclose(cpu,saved,rtol=1e-4,atol=2e-5)
        def neural():
            return ns['inverse_sample_standardization'](
                ns['predict_standardized_coefficients'](model,x,scaler,'cpu',batch_size=2048),scale)
        def full():
            inputs,scaling=preprocess(dim)
            return ns['inverse_sample_standardization'](
                ns['predict_standardized_coefficients'](model,inputs,scaler,'cpu',batch_size=2048),scaling)
        for function,name in [(lambda:preprocess(dim),'preprocessing'),(neural,'network_and_inverse'),(full,'end_to_end')]:
            for _ in range(3):function()
            for repeat in range(9):
                started=time.perf_counter();function();elapsed=time.perf_counter()-started
                rows.append(dict(dimension=dim,seed=seed,stage=name,repeat=repeat,n_cases=len(values),
                                 milliseconds=1000*elapsed,device='cpu',threads=4,batch_size=2048))
    atomic_csv(output/'inference_timing_repeats.csv',pd.DataFrame(rows))
    timing=pd.DataFrame(rows).groupby(['dimension','stage']).milliseconds.agg(
        median_ms='median',q25_ms=lambda x:x.quantile(.25),q75_ms=lambda x:x.quantile(.75)).reset_index()
    atomic_csv(output/'inference_timing_summary.csv',timing)
    return timing


def build_figures(output, manifest, data):
    metrics=pd.read_csv(output/'rmse_per_seed.csv')
    grouped=pd.read_csv(output/'grouped_rmse_per_seed.csv')
    group_summary=grouped.groupby(['dimension','axis','group','lower','upper','coefficient']).agg(
        n_cases=('n_cases','first'),n_seeds=('seed','size'),RMSE=('RMSE','mean'),RMSE_SD=('RMSE','std'),
        z_RMSE=('z_RMSE','mean'),z_RMSE_SD=('z_RMSE','std'),bias=('bias','mean')).reset_index()
    atomic_csv(output/'grouped_rmse_summary.csv',group_summary)
    fig,axes=plt.subplots(5,3,figsize=(12,14),sharey='row')
    # Display one paired seed, rather than pool repeated cases and call them
    # independent observations. Complete three-seed quantiles are in CSV.
    selected_seed=20260929
    errors={}
    for entry in manifest['runs']:
        if entry['seed']==selected_seed:
            errors[entry['dimension']]=np.load(Path(entry['path'])/'validation/predictions_original.npy')-data['validation']['coefficients']
    for j,name in enumerate(COEFFICIENT_NAMES):
        lower=min(np.min(e[:,j]) for e in errors.values())
        upper=max(np.max(e[:,j]) for e in errors.values())
        # Common bins per coefficient. All 10k cases retained, including tails.
        bins=np.linspace(lower,upper,81)
        for k,dim in enumerate((17,21,50)):
            ax=axes[j,k];e=errors[dim][:,j]
            ax.hist(e,bins=bins,color=COLORS[k],alpha=.85)
            ax.axvline(0,color='black',lw=.8)
            ax.set_title(f'{dim}D: {LABELS[j]} residual')
            ax.set_xlabel('Prediction - truth');ax.grid(alpha=.15)
            if k==0:ax.set_ylabel('Case count')
    fig.suptitle('Validation residual distributions: paired seed 20260929 (10,000 cases)',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.975));fig.savefig(output/'residual_distributions_5x3.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,axis in zip(axes.flat,['sigma0','beta_mu_over_sigma0','beta_sigma','xi0']):
        part=group_summary.loc[(group_summary.dimension==17)&(group_summary.axis==axis)]
        for j,name in enumerate(COEFFICIENT_NAMES):
            p=part.loc[part.coefficient==name].sort_values('group')
            ax.errorbar(p.group,p.z_RMSE,yerr=p.z_RMSE_SD,marker='o',capsize=3,label=LABELS[j])
        groups=part.drop_duplicates('group').sort_values('group')
        ax.set_xticks(groups.group,[f'{r.lower:g} to {r.upper:g}\n(n={int(r.n_cases):,})' for r in groups.itertuples()],fontsize=8)
        ax.set_title(axis);ax.set_ylabel('RMSE in training-target z units');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,fontsize=10,loc='upper center',bbox_to_anchor=(.5,.95))
    fig.suptitle('17D baseline: errors by known generating conditions (mean + seed SD)')
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(output/'grouped_errors_17d.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,dim in zip(axes,(17,21,50)):
        for entry in manifest['runs']:
            if entry['dimension']!=dim:continue
            history=pd.read_csv(Path(entry['path'])/'training_history.csv')
            color=COLORS[[20260929,20260930,20261001].index(entry['seed'])]
            ax.plot(history.epoch,history.validation_RMSE_eval,color=color,label=str(entry['seed']))
            ax.plot(history.epoch,history.train_RMSE_eval,color=color,ls='--',alpha=.7)
        ax.set_title(f'{dim} inputs');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Scaled coefficient RMSE');axes[0].legend(fontsize=8)
    fig.suptitle('Same end-of-epoch weights: solid = validation, dashed = train (no test)')
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=160);plt.close(fig)
    # Baseline group summary for slides: no arbitrary winner-driven bin choice.
    slide=group_summary.loc[(group_summary.dimension==17)&(group_summary.axis=='xi0')]
    atomic_csv(output/'beamer_shape_groups.csv',slide.pivot(index=['lower','upper','n_cases'],columns='coefficient',values='RMSE').reindex(columns=COEFFICIENT_NAMES).reset_index())
    # Compact slide subsets cover all four diagnostic axes. They overlap
    # across axes and are not a partition of the 10,000 cases.
    truth=np.asarray(data['validation']['coefficients'])
    sigma=np.exp(truth[:,2]);ratio=truth[:,1]/sigma
    masks={
        'sigma0 < 0.5':sigma<.5,
        'sigma0 >= 4':sigma>=4,
        'abs(beta_mu/sigma0) < 0.05':np.abs(ratio)<.05,
        'abs(beta_mu/sigma0) >= 0.2':np.abs(ratio)>=.2,
        'abs(beta_sigma) < 0.03':np.abs(truth[:,3])<.03,
        'abs(beta_sigma) >= 0.1':np.abs(truth[:,3])>=.1,
        'xi0 < -0.2':truth[:,4]<-.2,
        'xi0 >= 0.2':truth[:,4]>=.2,
    }
    subset_rows=[]
    for entry in manifest['runs']:
        if entry['dimension']!=17:continue
        e=np.load(Path(entry['path'])/'validation/predictions_original.npy')-truth
        for condition,mask in masks.items():
            subset_rows.append(dict(condition=condition,seed=entry['seed'],n_cases=int(mask.sum()),
                **dict(zip(COEFFICIENT_NAMES,np.sqrt(np.mean(e[mask]**2,axis=0))))))
    subsets=pd.DataFrame(subset_rows).groupby('condition',sort=False).agg(
        dict(n_cases='first',**{name:'mean' for name in COEFFICIENT_NAMES})).reset_index()
    atomic_csv(output/'beamer_grouped_conditions.csv',subsets)
    return group_summary


def notebook(output):
    def md(text):return dict(cell_type='markdown',metadata={},source=text.splitlines(True))
    def code(text):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True))
    cells=[md('# Cheng NN：17／21／50 維比較\n\n同一批 80,000 train 與 10,000 validation；test 未讀取。17 維沿用原本三個 seed，21／50 維各訓練三個 seed。只比較五係數，不評估 RL，不替換正式模型。'),
      code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / 'experiment.json').exists():\n    ROOT = Path("+repr(str(output))+ ")\nmanifest = __import__('json').loads((ROOT/'experiment.json').read_text(encoding='utf-8'))\nassert manifest['status'] == 'completed'\ndisplay(pd.read_csv(ROOT/'comparison_summary.csv').round(5))"),
      md('## 整體與學習曲線\n\n17 維是 11 個整體分位數＋三段 median/IQR（16/17/17 年）；21 維改為五段各 10 年；50 維是原順序標準化年最高溫。三者均採相同整段 median/IQR。整體 RMSE 在 train-only 目標 z 尺度比較，不混加原單位。三 seed 的 SD 不是資料抽樣的信賴區間。'),
      code("display(pd.read_csv(ROOT/'rmse_per_seed.csv').round(5))\ndisplay(Image(filename=str(ROOT/'learning_curves.png')))"),
      md('## 分組誤差與分布\n\n用事先固定的 sigma0、beta_mu/sigma0、beta_sigma、xi0 區間分组，各組五係數皆報原尺度 RMSE 與標準化 RMSE。大 sigma 的攝氏度誤差增加，不等於標準化估計變差。直方圖只顯示第一個共同 seed，所有 seed 的分位數摘要另存 CSV，不把重複案例當成獨立樣本。'),
      code("display(pd.read_csv(ROOT/'grouped_rmse_summary.csv').round(5))\ndisplay(Image(filename=str(ROOT/'grouped_errors_17d.png')))\ndisplay(Image(filename=str(ROOT/'residual_distributions_5x3.png')))"),
      md('## GEV 支持範圍\n\n有效分布參數與觀測值落在支持範圍內是不同檢查。xi 不必為正。超出訓練 xi 區間單獨標記，不當成分布無效。所有案例仍納入係數 RMSE，沒有刪掉支持範圍失敗案例。'),
      code("display(pd.read_csv(ROOT/'gev_validity_summary.csv'))\ndisplay(pd.read_csv(ROOT/'support_failure_cases.csv'))"),
      md('## 推論耗時\n\nCPU 4 threads，每次 10,000 案例、batch=2048；每模型暖身三次後計時九次。端到端包含整段 median/IQR、輸入特徵、神經網路和係數逆轉換，不含讀檔或模型載入。'),
      code("display(pd.read_csv(ROOT/'inference_timing_summary.csv').round(3))")]
    atomic_json(output/'cheng_NN_input_comparison.ipynb',dict(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=4))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--skip-timing',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    manifest=read_json(output/'experiment.json')
    if manifest['status']!='completed' or len(manifest['runs'])!=9 or any(r['status']!='completed' for r in manifest['runs']):
        raise ValueError('Wait until all nine paired models are complete.')
    expected={(dim,seed) for dim in (17,21,50) for seed in (20260929,20260930,20261001)}
    if {(r['dimension'],r['seed']) for r in manifest['runs']}!=expected:
        raise ValueError('Runs are not paired across all three seeds.')
    for path,digest in manifest['source_hashes'].items():
        if file_sha256(path)!=digest:
            raise ValueError(f'Training source changed: {path}')
    for split,files in manifest['dataset_files_sha256'].items():
        for name,digest in files.items():
            if file_sha256(Path(manifest['data_directory'])/split/name)!=digest:
                raise ValueError(f'Source data changed: {split}/{name}')
    ns,data=load_data(manifest['data_directory'])
    build_figures(output,manifest,data)
    if not args.skip_timing:benchmark(output,manifest,ns,data)
    notebook(output)
    atomic_json(output/'report_metadata.json',dict(
        status='completed',n_models=9,n_train=80000,n_validation=10000,test_used=False,
        all_source_data_hashes_verified=True,
        report_source_sha256=file_sha256(Path(__file__)),
        study_manifest_sha256=file_sha256(output/'experiment.json')))
    print('Diagnostic tables, figures, timing and notebook are ready.')


if __name__=='__main__':main()
