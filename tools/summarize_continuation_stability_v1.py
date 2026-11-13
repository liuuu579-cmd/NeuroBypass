"""Interpret Stage27 and separate sequence-shape numerics from window variation."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from continuation_stability_core_v1 import estimate
from full_context_experiment_v1 import read_lines, write


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]; out = root / 'runs/stage27_continuation_stability_20261113_01_main'
    cfg = load(out / 'protocol.json'); summary = load(out / 'summary.json'); audit = load(out / 'independent_audit.json')
    assert audit['passed']; ids = cfg['ids']; old = Path(cfg['old_run'])
    old_labels = {r['sample']: r for rank in range(6) for r in read_lines(old / f'private_labels_{rank:03d}.jsonl')}
    new = {r['sample']: r for rank in range(4) for r in read_lines(out / f'private_labels_{rank:03d}.jsonl')}
    old32 = np.asarray([old_labels[i]['kl'] for i in ids]); costs = np.asarray([new[i]['kl'] for i in ids])
    first = costs[:, 0]; old_choice = old32.argmin(1); new_choice = first.argmin(1)
    numerical = {'same_first32_tokens': True, 'mean_absolute_kl_difference': float(np.abs(old32 - first).mean()),
        'maximum_absolute_kl_difference': float(np.abs(old32 - first).max()),
        'winner_agreement': float(np.mean(old_choice == new_choice)),
        'rank_spearman': estimate([spearmanr(a, b).statistic for a, b in zip(old32, first)]),
        'old_choice_regret_under_new_first32': estimate(first[np.arange(128), old_choice] - first.min(1)),
        'old32_choice_improvement_on_later96': estimate((costs[:, 1:, 1] - costs[np.arange(128)[:, None], np.arange(1, 4)[None], old_choice[:, None]]).mean(1)),
        'interpretation': 'Short-vs-long FP16 computation shapes differ despite identical prefix tokens; direct old-length replays and same-shape future-causality tests passed. This is distinct from changing continuation windows.'}
    write(out / 'original_window_numerical_comparison.json', numerical)
    by_key = {(r['policy'], str(r['window']), r['metric']): r for r in summary['policy_transfer']}
    transfer_first = by_key['first32_future_oracle', '0', 'kl']
    transfer_later = by_key['first32_future_oracle', 'later96', 'kl']
    transfer_all = by_key['first32_future_oracle', 'all128', 'kl']
    oracle_all = by_key['full128_future_oracle', 'all128', 'kl']
    first_pairs = [p for p in summary['stability']['window_pairs'] if p['windows'][0] == 0]
    interpretation = {'numerical_control': numerical, 'first_window_to_later': first_pairs,
        'first32_oracle_gain_first32': transfer_first['improvement'],
        'first32_oracle_gain_later96': transfer_later['improvement'],
        'first32_oracle_gain_all128': transfer_all['improvement'],
        'full128_oracle_gain_all128': oracle_all['improvement'],
        'conclusion': 'Short-window route winners are weakly persistent along natural continuations. First32 future-informed mask has no stable later96 gain over fixed strong static. Supports a horizon-alignment concern, not a proof of random label noise or input-only unlearnability.',
        'no_retraining_or_config_selection': True, 'limits': cfg['limits']}
    write(out / 'diagnostic_interpretation.json', interpretation)
    text = ['# 阶段27结论：短续写窗口的最优组合难以代表后续整段输出', '',
        '本轮完成128条旧训练文档的诊断，保留原完整提示词，在同一文档的后续128个token上比较四个32-token窗口。每次评分使用相同的8个K20完整组合，组合从提示词到续写结束始终固定，采用真实教师联合KL。没有更新OPT、路由器或候选库。', '',
        '## 主要结果', '', '| 首32-token窗口与后续窗口比较 | 最优组合一致率 | 8组合排名Spearman | 扣除边际频率后的Kappa |', '|---|---:|---:|---:|']
    for row in first_pairs:
        text.append(f"| 窗口0与窗口{row['windows'][1]} | {100 * row['winner_agreement']:.2f}% | {row['within_document_route_spearman']['mean']:.3f} | {row['cohen_kappa']:.3f} |")
    text += ['', '最优组合变化不只来自几乎相同的代价：全部窗口中，最优与次优KL差小于0.01的情况占' + f"{100 * summary['stability']['fraction_winner_margin_below_tau001']:.2f}%" + '，差值中位数为' + f"{summary['stability']['best_second_gap_quantiles'][1]:.4f}" + '。不过最优索引也不是唯一标准，报告同时列出完整排名、代价后悔值、0.01范围内近优组合的重叠，以及有明显代价差的成对排名一致率。', '',
        '| 固定一个组合的策略 | 首32 token KL | 后96 token KL | 全128 token KL |', '|---|---:|---:|---:|']
    for policy, label in (('strong_static', '强静态索引1'), ('first32_future_oracle', '知道首32真实KL后选最优，仅诊断'),
        ('full128_future_oracle', '知道全部128真实KL后选最优，仅诊断'), ('embedding_gate_mean3', '旧embedding路由，路径已固定'), ('hidden2_gate_mean3', '旧两层语义路由，路径已固定')):
        values = [by_key[policy, w, 'kl']['mean'] for w in ('0', 'later96', 'all128')]
        text.append(f"| {label} | " + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lo, hi = transfer_later['improvement']['ci95']
    text += ['', f"首32窗口真实最优，在首窗上相对静态改善{transfer_first['improvement']['mean']:.6f}；同一个组合应用于后96 token，改善变为{transfer_later['improvement']['mean']:+.6f}，探索性95%区间[{lo:+.6f}, {hi:+.6f}]。即使已经知道首32个未来token，也没有建立后续96 token上的可靠优势。", '',
        f"知道整个128-token未来后选固定组合，仍有{oracle_all['improvement']['mean']:.6f}的平均KL改善空间。这只说明同库存在长段质量空间，不是输入路由器已能取得该收益。用其余96个未来token选组合、评估留下的32 token，平均改善约{by_key['other96_future_oracle', 'all128', 'kl']['improvement']['mean']:.6f}，其区间仍跨零，也不能直接证明多窗口监督已经足够。", '',
        '## 排除数值重算作为主要解释', '',
        f"同样的首32个token，旧短序列与新长序列的KL平均绝对差为{numerical['mean_absolute_kl_difference']:.6f}，最大差为{numerical['maximum_absolute_kl_difference']:.6f}；最优组合一致率{100 * numerical['winner_agreement']:.2f}%，排名相关{numerical['rank_spearman']['mean']:.4f}。用旧32-token最优组合而非新首窗最优组合，后96窗口的结论仍一致。数值敏感性需要保留，但不能主要解释跨窗口只有16%—25%的最优组合一致率。", '',
        '四个worker都在原短长度上重现了既有32-token联合KL；同一新长序列重复评分逐token一致，改变后96个token也不改变首32个token的评分。以上用于区分FP16序列形状敏感性、可复现性和因果边界，不将它们当作随机标签噪声。', '',
        '## 对训练方向的含义', '',
        '当前训练目标只覆盖一小段真实续写，但部署路径需要服务整个回答。本轮支持“短窗口局部最优与整段固定路径目标可能不匹配”的解释；不能据此断言标签错误、不可学习，或者只要平均窗口就能超过静态。不同续写内容和逐渐增长的教师强制上下文均可能引起排名变化。', '',
        '若继续推进，建议先小试多窗口平均联合KL或多条教师续写平均KL，并保留同预算的常量对照。要严格区分同一提示词下的未来采样波动，需要从同一提示词独立采样多条教师续写，再做分支间的选层收益迁移；本轮连续窗口不满足独立同分布假设。本轮没有启动该训练或采样实验。', '',
        '## 执行与限制', '',
        '原800条训练文档中351条满足原提示词之后有128个同文档token且不超2048上下文；按事先固定种子不看KL抽128条，每个正文/开头分组最多一条。实际提示词长度为64、128、256、512、1024，分别46、33、28、15、6条。很多短文档与长提示词被排除，因此不能直接外推到全部800条或聊天任务。原100条开发文档未参与。', '',
        f"独立审计重建抽样与全部指标，校验{audit['per_token_metric_values_verified']}个逐token指标值，额外在4条长提示词上重放32个完整剪层组合，逐token结果完全一致。{audit['preserved_files']}个旧文件验哈希保全。使用4张GPU、42个CPU核心；没有新增吞吐、显存或恢复训练实验。", '',
        '候选库历史上用过部分原训练文档，旧外折文档也已多轮查看，本轮是开发诊断。探索性区间只按文档重采样，不包含模型训练/候选筛选/折间依赖，未作多重比较校正。所有future_oracle策略读取真实未来，仅为诊断，不是线上可用结果；mean3为独立种子质量均值，不是集成部署。权重、语料、token和逐文档记录不上传。', '',
        f"[完整结果](stage27_continuation_stability_20261113_01_main_results.zh-CN.md) · [数值敏感性对照](../{out.relative_to(root)}/original_window_numerical_comparison.json) · [独立审计](../{out.relative_to(root)}/independent_audit.json)"]
    with (root / 'analysis/STAGE27_CONTINUATION_STABILITY_CONCLUSION_20261113.zh-CN.md').open('x') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps(interpretation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
