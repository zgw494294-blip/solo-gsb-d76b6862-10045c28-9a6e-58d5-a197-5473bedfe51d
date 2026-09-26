"""首次故障时刻分布：共享事件状态马尔可夫链上的吸收概率动态规划。

t=0 各事件按先验 p 独立生成状态；此后每一步，给定上一步状态，各事件
独立转移：正常（false）事件以 fail 概率故障，故障（true）事件以
recover 概率恢复。每步用共享事件状态按 AND/OR/NOT 门求顶事件取值。

顶事件首次发生即吸收：概率动态规划只携带尚未触发顶事件的状态，当步
触发的质量进入 first_hit_distribution[t]，不再向后传播；过程结束仍
存活的质量即 no_hit_probability。事件数 ≤ 8，完整状态空间 ≤ 256，
逐步精确转移，全部用 Fraction 精确有理数运算。

各事件给定上步状态后独立，联合转移矩阵是各事件 2×2 转移矩阵的克罗
内克积，故对分布向量逐事件作用即可，无需显式构造 2^n×2^n 矩阵，
每步仅 O(n·2^n) 次分数运算。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple


def _make_state_evaluator(
    event_index: Dict[str, int],
    gate_ops: Dict[str, str],
    gate_inputs: Dict[str, List[str]],
):
    """返回 evaluate(state_bits, node_id)：在给定事件状态下递归求值门。

    同一状态内各门的求值结果做记忆化（门依赖是无环 DAG，已在校验阶段
    保证）；同一事件 id 在故障树中的多处引用共享同一个状态位。
    """

    def evaluate(state: int, node_id: str, memo: Dict[str, bool]) -> bool:
        idx = event_index.get(node_id)
        if idx is not None:
            return bool((state >> idx) & 1)
        cached = memo.get(node_id)
        if cached is not None:
            return cached
        op = gate_ops[node_id]
        inputs = gate_inputs[node_id]
        if op == "NOT":
            value = not evaluate(state, inputs[0], memo)
        elif op == "AND":
            value = all(evaluate(state, ref, memo) for ref in inputs)
        else:  # OR
            value = any(evaluate(state, ref, memo) for ref in inputs)
        memo[node_id] = value
        return value

    return evaluate


def first_hit_distribution(
    event_ids: List[str],
    probs: List[Fraction],
    gate_ops: Dict[str, str],
    gate_inputs: Dict[str, List[str]],
    top: str,
    fail_probs: List[Fraction],
    recover_probs: List[Fraction],
    steps: int,
) -> Tuple[List[Fraction], Fraction]:
    """求顶事件首次发生时刻分布。

    event_ids 已按字典序排序，第 i 位对应第 i 个事件；probs /
    fail_probs / recover_probs 同序，分别给出 t=0 先验、每步正常→故障
    与故障→正常的概率。返回 (长度 steps+1 的首次命中分布, 全程未命中
    概率)，下标 t=0~steps。
    """
    n = len(event_ids)
    n_states = 1 << n
    event_index = {eid: i for i, eid in enumerate(event_ids)}
    evaluate = _make_state_evaluator(event_index, gate_ops, gate_inputs)

    # 顶事件取值只是事件状态的静态布尔函数，对全部状态预计算一次
    top_hits = [evaluate(s, top, {}) for s in range(n_states)]

    # t=0：各事件按 p 独立生成初始状态分布
    alive: List[Fraction] = [Fraction(0)] * n_states
    for s in range(n_states):
        weight = Fraction(1)
        for i, p in enumerate(probs):
            weight *= p if (s >> i) & 1 else (Fraction(1) - p)
        alive[s] = weight

    first_hit: List[Fraction] = [Fraction(0)] * (steps + 1)

    for t in range(steps + 1):
        # 吸收：本步顶事件发生的状态即“首次在 t 发生”，质量移出存活分布
        for s in range(n_states):
            if top_hits[s] and alive[s] != 0:
                first_hit[t] += alive[s]
                alive[s] = Fraction(0)
        if t == steps:
            break

        # 转移：给定上步状态各事件独立。逐事件作用其 2×2 转移矩阵：
        #   0 → 1 以 fail、0 → 0 以 (1−fail)；
        #   1 → 0 以 recover、1 → 1 以 (1−recover)。
        for i in range(n):
            bit = 1 << i
            fail = fail_probs[i]
            recover = recover_probs[i]
            nxt = [Fraction(0)] * n_states
            for s, mass in enumerate(alive):
                if mass == 0:
                    continue
                if s & bit:
                    # 故障事件：保持故障或恢复
                    nxt[s] += mass * (Fraction(1) - recover)
                    nxt[s ^ bit] += mass * recover
                else:
                    # 正常事件：保持正常或故障
                    nxt[s] += mass * (Fraction(1) - fail)
                    nxt[s ^ bit] += mass * fail
            alive = nxt

    # 最后一步仍存活的质量即 steps 步内顶事件始终未发生
    no_hit = sum(alive, Fraction(0))
    return first_hit, no_hit
