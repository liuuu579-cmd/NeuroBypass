"""Append-only Stage29 interpretation and publication-resume manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--allowlist', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.run.resolve()
    audit = load(args.audit)
    assert audit['passed'] and load(out/'verification.json')['passed']
    s = load(out/'summary.json')
    def value(task, name, k):
        rows = [r for r in s['quality'] if (r['task'], r['policy'], r['k']) == (task, name, k)]
        field = 'conditional_ppl' if task == 'wikitext_conditional' else 'accuracy' if task == 'winogrande' else 'accuracy_norm'
        return sum(r[field] for r in rows)/len(rows)
    def fmt(task, val):
        return f'{val:.3f}' if task == 'wikitext_conditional' else f'{val*100:.3f}%'
    lines = ['# 阶段29：完整横评结论与发布续传', '',
        '**完整评测和审计已结束。目前有局部优势，尚不支持“动态全面优于其他方法”的结论。**', '',
        '冻结权重的OPT-13B，整层剪枝K12/16/20/24，原始层0/1保留。共13,893条题目/条件窗口、',
        '778,008行含跨卡完整模型复核的评分，去重后736,329行。独立CPU审计重算212组聚合指标与8项K20配对区间，',
        '核对搜索轨迹、校准TRAIN身份、源码及检查点哈希；3,717个既有文件重新核对未变。', '',
        '## K20候选库动态与新增文献基线', '',
        '动态固定使用Stage19的bank8/256条TRAIN版本、三个既有种子，非按本轮结果重新挑选。',
        '下表均为三个独立种子的指标平均，强静态为一个在TRAIN800上预先选择的固定组合；不是推理集成。', '',
        '| 任务 | bank8动态 | EvoPress适配 | 我们的强静态 |', '|---|---:|---:|---:|']
    tasks = ['hellaswag', 'piqa', 'winogrande', 'wikitext_conditional']
    for task in tasks:
        vals = [fmt(task, value(task, name, 20)) for name in
            ['codebook8_train256_prompt', 'evopress_whole_opt', 'static_refined_bank_train800']]
        lines.append('| '+task+' | '+' | '.join(vals)+' |')
    lines += ['', 'PIQA/HellaSwag为acc_norm，Wino为acc（越高越好），Wiki为条件PPL（越低越好）。',
        'Wiki每个种子内部按token加权，再对种子的PPL取算术平均；不等于先平均NLL再取指数。', '',
        '与EvoPress比较，bank8动态在三项选择题均值上均领先。HellaSwag差异为+1.026个百分点，',
        '按题配对95%区间[+0.485,+1.547]个百分点；PIQA为+1.161个百分点[-0.199,+2.575]，',
        'Wino为+1.289个百分点[-1.106,+3.525]，后二者区间跨零。Wiki条件PPL略差，窗口平均NLL的差异区间也跨零。',
        '这些区间未校正多重比较，只描述当前固定的三个训练/搜索种子的题目重采样不确定性。', '',
        '与自己的强静态比较，动态的HellaSwag低1.268个百分点（区间[-1.700,-0.833]），Wiki条件PPL也更差；',
        'PIQA高0.272个百分点、Wino低0.947个百分点，二者区间跨零。该强静态使用额外TRAIN搜索，',
        '不是等离线成本对照；保留它是为了避免把候选组合本身的质量归因为输入适配。', '',
        '## 其他剪层预算', '',
        '| K | Hella动态m6 / EvoPress | PIQA动态m6 / EvoPress | Wino动态m6 / EvoPress | Wiki条件PPL动态m6 / EvoPress |',
        '|---:|---:|---:|---:|---:|']
    for k in (12, 16, 20, 24):
        vals = [fmt(t, value(t, 'm06', k))+' / '+fmt(t, value(t, 'evopress_whole_opt', k)) for t in tasks]
        lines.append('| '+str(k)+' | '+' | '.join(vals)+' |')
    lines += ['', 'm6在K12四项均值都低于EvoPress；K16只在Wino均值上占优，K20也只在Wino占优。',
        'K24的Hella和PIQA有正向信号，但Wino较差，Wiki条件PPL恶化到约420，深剪质量没有整体保持。',
        '所有预算下都不能合并成“动态普遍更好”。bank8只在K20加入本次横评，不能从其K20结果推断其他K。', '',
        '## 结论的边界与下一步', '',
        '1. 本次EvoPress为预算受限OPT整层适配，64代低于论文脚本约224–267代，不能声称已经击败完整官方配方。',
        '   原生全序列KL与我们静态的32位置续写KL也不同。SLEB/ShortGPT/Shortened为既有冻结OPT适配，均未做本轮恢复微调。',
        '2. 任务人口此前已被审阅，Wiki使用条件窗口而非标准rolling PPL；这不是一套全新终测。',
        '3. 三项选择题均值领先提供了继续研究K20 bank8的依据，但输入适配贡献仍需同候选库固定/无输入对照；',
        '   不能只与较弱论文适配结果比较就宣称动态机制有效。当前横评没有额外评测无输入常数路由检查点。',
        '4. 下一步建议先补齐EvoPress论文量级搜索与同候选库无输入对照，再考虑等预算恢复微调；保持现有动态检查点冻结。',
        '   本次核查没有启动新GPU训练或搜索。', '',
        '## 发布状态说明', '',
        '原流水线的计算阶段成功退出。发布阶段因summary.json在JSON转义后超过v8发布工具单文件内联阈值而停止。',
        '新增publish_text_additions_v9.py对较大文本使用独立Git blob，然后只把SHA写入有界tree请求；',
        '保留原文件字节和哈希，旧程序、报告和输出均不修改。发布成功与否由新的publication_v2收据确认，不能混同为计算失败。', '',
        '[完整逐方法结果](STAGE29_RESULTS_20261113.zh-CN.md) · ',
        '[方案](STAGE29_LITERATURE_COMPARISON_PROTOCOL_20261113.zh-CN.md) · ',
        '[独立审计](stage29_literature_final_audit_20261114.json)', '',
        f'本地运行目录：`{out.relative_to(root)}`。']
    with args.report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    paths = load(root/'analysis/stage29_literature_20261113_01_main_public_files.json')
    paths += [str(args.report.resolve().relative_to(root)), str(args.audit.resolve().relative_to(root)),
        'tools/audit_literature_completed_v1.py', 'tools/summarize_literature_findings_v1.py',
        'tools/publish_text_additions_v9.py']
    assert len(paths) == len(set(paths))
    write(args.allowlist, paths)
    print(json.dumps({'report': str(args.report), 'public_files': len(paths)}), flush=True)


if __name__ == '__main__':
    main()
