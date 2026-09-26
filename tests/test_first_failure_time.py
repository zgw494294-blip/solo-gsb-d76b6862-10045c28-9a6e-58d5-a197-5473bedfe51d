"""POST /v1/first-failure-time：首次故障时刻分布的语义、精确性与 422 校验。

参考实现独立构造 2^n×2^n 的显式状态转移矩阵做向量推进（实现侧则用
逐事件 2×2 转移矩阵的克罗内克积作用，避免显式矩阵），两侧交叉验证。
"""

import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------- 显式全转移矩阵参考实现 ----------

def brute_first_hit(ev_specs, gate_specs, top, steps, transitions):
    """显式构造全部状态间的联合转移概率矩阵推进，返回
    ([P(首次在 t 发生) for t in 0..steps], 全程未发生概率)。
    """
    ev_ids = sorted(ev_specs)
    n = len(ev_ids)
    size = 1 << n
    probs = [Fraction(ev_specs[e]) for e in ev_ids]
    fails = [Fraction(transitions[e]["fail"]) for e in ev_ids]
    recovers = [Fraction(transitions[e]["recover"]) for e in ev_ids]
    gates = {g[0]: (g[1], g[2]) for g in gate_specs}

    def evaluate(state, node):
        if node in ev_ids:
            return bool((state >> ev_ids.index(node)) & 1)
        op, inputs = gates[node]
        vals = [evaluate(state, i) for i in inputs]
        if op == "AND":
            return all(vals)
        if op == "OR":
            return any(vals)
        return not vals[0]

    hits = [evaluate(s, top) for s in range(size)]

    def trans_prob(s, nxt):
        # 给定上步状态各事件独立转移，联合概率为各事件因子之积
        w = Fraction(1)
        for i in range(n):
            b = (s >> i) & 1
            nb = (nxt >> i) & 1
            if b:
                w *= (1 - recovers[i]) if nb else recovers[i]
            else:
                w *= fails[i] if nb else (1 - fails[i])
        return w

    matrix = [
        [trans_prob(s, nxt) for nxt in range(size)] for s in range(size)
    ]

    # t=0 各事件按先验独立生成
    dist = [Fraction(0)] * size
    for s in range(size):
        w = Fraction(1)
        for i in range(n):
            w *= probs[i] if (s >> i) & 1 else (1 - probs[i])
        dist[s] = w

    first_hit = [Fraction(0)] * (steps + 1)
    for t in range(steps + 1):
        # 顶事件首次发生即吸收：质量移出存活分布，不再向后传播
        for s in range(size):
            if hits[s]:
                first_hit[t] += dist[s]
                dist[s] = Fraction(0)
        if t == steps:
            break
        nxt_dist = [Fraction(0)] * size
        for s, mass in enumerate(dist):
            if mass:
                for nxt, q in enumerate(matrix[s]):
                    if q:
                        nxt_dist[nxt] += mass * q
        dist = nxt_dist
    return first_hit, sum(dist, Fraction(0))


def f2s(f):
    return f"{f.numerator}/{f.denominator}"


def make_body(ev_specs, gate_specs, top, steps, transitions):
    return {
        "events": [{"id": k, "p": v} for k, v in ev_specs.items()],
        "gates": [
            {"id": i, "op": op, "inputs": ins}
            for i, op, ins in gate_specs
        ],
        "top": top,
        "steps": steps,
        # 原样透传，允许构造缺键/多键的非法项以覆盖 422 路径
        "transitions": {e: dict(v) for e, v in transitions.items()},
    }


def uniform_transitions(ev_ids, fail="1/2", recover="1/2"):
    return {e: {"fail": fail, "recover": recover} for e in ev_ids}


def check_against_brute(ev_specs, gate_specs, top, steps, transitions):
    body = make_body(ev_specs, gate_specs, top, steps, transitions)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 200, (r.text, body)
    exp_hit, exp_no = brute_first_hit(
        ev_specs, gate_specs, top, steps, transitions
    )
    data = r.json()
    assert len(data["first_hit_distribution"]) == steps + 1
    assert data["first_hit_distribution"] == [f2s(p) for p in exp_hit]
    assert data["no_hit_probability"] == f2s(exp_no)
    # 首次命中各时刻概率与未命中概率之和恒为 1
    total = sum(
        (Fraction(s) for s in data["first_hit_distribution"]),
        Fraction(data["no_hit_probability"]),
    )
    assert total == 1
    return data


# ---------- 基本语义（手算）----------

def test_single_event_top_hand_computed():
    # 单事件即顶事件：p=1/2，fail=1/4，recover=1/3，steps=2
    # t=0：1/2 命中；存活 a=0 质量 1/2
    # t=1：1/2·1/4=1/8 命中；存活 1/2·3/4=3/8
    # t=2：3/8·1/4=3/32 命中；未命中 3/8·3/4=9/32
    transitions = {"a": {"fail": "1/4", "recover": "1/3"}}
    body = make_body({"a": "1/2"}, [], "a", 2, transitions)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "first_hit_distribution": ["1/2", "1/8", "3/32"],
        "no_hit_probability": "9/32",
    }


def test_and_gate_two_events_hand_computed():
    # top = a AND b；p_a=1/2, p_b=1/3；fail=recover=1/2；steps=2
    ev = {"a": "1/2", "b": "1/3"}
    gates = [("top", "AND", ["a", "b"])]
    data = check_against_brute(
        ev, gates, "top", 2, uniform_transitions(["a", "b"])
    )
    # t=0：P(a=b=1)=1/6；存活 00=1/3、10=1/6、01=1/3（合计 5/6）。
    # 每步各事件独立地以 1/2 翻转/保持，故从任一状态到任一目标状态
    # 的概率都是 1/4：
    #   t=1 到 11：(1/3+1/6+1/3)·1/4 = 5/24；存活三态各 5/24
    #   t=2 到 11：3·(5/24·1/4) = 5/32；未命中 5/8−5/32 = 15/32
    assert data["first_hit_distribution"] == ["1/6", "5/24", "5/32"]
    assert data["no_hit_probability"] == "15/32"


def test_constant_true_top_hits_at_t0():
    # top = a OR NOT a 恒真：t=0 必命中，其后全为 0，未命中为 0
    ev = {"a": "3/7"}
    gates = [("na", "NOT", ["a"]), ("top", "OR", ["a", "na"])]
    data = check_against_brute(
        ev, gates, "top", 3, uniform_transitions(["a"])
    )
    assert data["first_hit_distribution"] == ["1/1", "0/1", "0/1", "0/1"]
    assert data["no_hit_probability"] == "0/1"


def test_constant_false_top_never_hits():
    # top = a AND NOT a 恒假：各时刻命中均为 0，未命中为 1（本接口不
    # 因顶事件恒假而报错——分布仍然良定义）
    ev = {"a": "3/7"}
    gates = [("na", "NOT", ["a"]), ("top", "AND", ["a", "na"])]
    data = check_against_brute(
        ev, gates, "top", 3, uniform_transitions(["a"])
    )
    assert data["first_hit_distribution"] == ["0/1"] * 4
    assert data["no_hit_probability"] == "1/1"


def test_fail_zero_keeps_normal_events_normal():
    # fail=0 且 t=0 顶事件（a）未发生后，正常事件永不故障：
    # p=1/3 ⇒ t=0 命中 1/3，未命中 2/3
    transitions = {"a": {"fail": "0/1", "recover": "1/1"}}
    body = make_body({"a": "1/3"}, [], "a", 4, transitions)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "first_hit_distribution": ["1/3", "0/1", "0/1", "0/1", "0/1"],
        "no_hit_probability": "2/3",
    }


def test_recover_one_all_failed_events_recover():
    # p=1/1 ⇒ t=0 必为故障；t=1 起恢复（recover=1/1）且 fail=0，
    # 顶事件（a）永不复现：t=0 命中 1，其余 0，未命中 0
    transitions = {"a": {"fail": "0/1", "recover": "1/1"}}
    body = make_body({"a": "1/1"}, [], "a", 2, transitions)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "first_hit_distribution": ["1/1", "0/1", "0/1"],
        "no_hit_probability": "0/1",
    }


def test_not_gate_top():
    # 顶事件为 NOT a：a 正常时顶事件发生
    ev = {"a": "2/5"}
    gates = [("top", "NOT", ["a"])]
    transitions = {"a": {"fail": "1/4", "recover": "3/8"}}
    data = check_against_brute(
        ev, gates, "top", 3, transitions
    )
    # t=0：P(a=0)=3/5
    assert data["first_hit_distribution"][0] == "3/5"


def test_unreferenced_event_does_not_change_result():
    # b 不被顶层引用：结果与只声明 a 时一致
    tr = {"a": {"fail": "1/3", "recover": "2/7"},
          "b": {"fail": "5/9", "recover": "1/6"}}
    data = check_against_brute(
        {"a": "1/2", "b": "2/3"}, [], "a", 3, tr
    )
    # 单事件 a 参考：p=1/2，fail=1/3
    exp_hit, exp_no = brute_first_hit(
        {"a": "1/2"}, [], "a", 3,
        {"a": tr["a"]},
    )
    assert data["first_hit_distribution"] == [f2s(p) for p in exp_hit]
    assert data["no_hit_probability"] == f2s(exp_no)


def test_shared_event_reference():
    # OR 门重复引用 a：共享同一状态，等价于 top=a
    ev = {"a": "2/5", "b": "1/4"}
    gates = [("g", "OR", ["a", "a", "b"])]
    tr = {
        "a": {"fail": "1/3", "recover": "1/2"},
        "b": {"fail": "2/3", "recover": "1/4"},
    }
    check_against_brute(ev, gates, "g", 4, tr)


def test_steps_boundaries_one_and_ten():
    ev = {"a": "1/2", "b": "1/3"}
    gates = [("top", "OR", ["a", "b"])]
    tr = uniform_transitions(["a", "b"], fail="1/5", recover="2/5")
    data1 = check_against_brute(ev, gates, "top", 1, tr)
    assert len(data1["first_hit_distribution"]) == 2
    data10 = check_against_brute(ev, gates, "top", 10, tr)
    assert len(data10["first_hit_distribution"]) == 11


def test_eight_events_ten_steps_performance():
    # 上界规模：8 事件、10 步，256 个状态，必须快速精确完成
    ev = {f"e{i}": f"{i+1}/{i+3}" for i in range(8)}
    gates = []
    prev = "e0"
    for i in range(1, 8):
        gid = f"g{i}"
        gates.append((gid, "OR" if i % 2 else "AND", [prev, f"e{i}"]))
        prev = gid
    tr = {
        f"e{i}": {"fail": f"{i+1}/{i+5}", "recover": f"{i+2}/{i+7}"}
        for i in range(8)
    }
    data = check_against_brute(ev, gates, prev, 10, tr)
    assert len(data["first_hit_distribution"]) == 11


# ---------- 随机电路穷举（显式矩阵）对照 ----------

def test_random_circuits_against_matrix_reference():
    rng = random.Random(20260928)
    for _ in range(60):
        n = rng.randint(1, 5)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 9)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"
        transitions = {}
        for e in ev_ids:
            df = rng.randint(1, 9)
            dr = rng.randint(1, 9)
            transitions[e] = {
                "fail": f"{rng.randint(0, df)}/{df}",
                "recover": f"{rng.randint(0, dr)}/{dr}",
            }
        gates = []
        available = list(ev_ids)
        for j in range(rng.randint(0, min(12, 3 * n))):
            gid = f"g{j}"
            if len(available) < 2:
                op = "NOT"
            else:
                op = rng.choice(["AND", "OR", "NOT"])
            if op == "NOT":
                inputs = [rng.choice(available)]
            else:
                k = rng.randint(2, min(8, len(available)))
                inputs = rng.sample(available, k)
            gates.append((gid, op, inputs))
            available.append(gid)
        top = rng.choice(available)
        steps = rng.randint(1, 6)
        check_against_brute(
            ev_specs, gates, top, steps, transitions
        )


# ---------- 422 校验 ----------

BASE_EV = {"a": "1/2", "b": "1/3"}
BASE_TR = {
    "a": {"fail": "1/4", "recover": "1/2"},
    "b": {"fail": "1/5", "recover": "2/3"},
}
BASE_GATES = [("top", "OR", ["a", "b"])]


def post_body(**overrides):
    body = make_body(BASE_EV, BASE_GATES, "top", 2, BASE_TR)
    body.update(overrides)
    return client.post("/v1/first-failure-time", json=body)


def test_missing_transition_entry():
    body = make_body(BASE_EV, BASE_GATES, "top", 2, {"a": BASE_TR["a"]})
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "transitions" in detail and "缺少" in detail and "'b'" in detail


def test_extra_transition_entry():
    tr = dict(BASE_TR)
    tr["x"] = {"fail": "1/2", "recover": "1/2"}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "transitions" in detail and "键" in detail and "'x'" in detail


def test_gate_id_in_transitions_is_extra():
    # 门 id 不是事件 id：transitions 只接受事件键
    tr = dict(BASE_TR)
    tr["top"] = {"fail": "1/2", "recover": "1/2"}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "'top'" in r.json()["detail"]


def test_missing_and_extra_reported_together():
    tr = {"a": BASE_TR["a"], "x": {"fail": "1/2", "recover": "1/2"}}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "缺少" in detail and "'b'" in detail
    assert "键" in detail and "'x'" in detail


def test_invalid_fail_probability():
    tr = {"a": {"fail": "3/2", "recover": "1/2"},
          "b": {"fail": "1/5", "recover": "2/3"}}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "fail" in detail and "'a'" in detail and "概率" in detail


def test_invalid_recover_probability():
    tr = {"a": {"fail": "1/4", "recover": "x"},
          "b": {"fail": "1/5", "recover": "2/3"}}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "recover" in detail and "'a'" in detail


def test_probability_denominator_too_large():
    tr = {
        "a": {"fail": "1/1000001", "recover": "1/2"},
        "b": {"fail": "1/5", "recover": "2/3"},
    }
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "10^6" in r.json()["detail"]


def test_multiple_bad_probabilities_reported_together():
    tr = {"a": {"fail": "2/1", "recover": "3/1"},
          "b": {"fail": "ok", "recover": "2/3"}}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "fail" in detail and "recover" in detail
    assert "'a'" in detail and "'b'" in detail


def test_transition_entry_extra_field():
    body = post_body()
    assert body.status_code == 200, body.text
    tr = {
        "a": {"fail": "1/4", "recover": "1/2", "extra": 1},
        "b": {"fail": "1/5", "recover": "2/3"},
    }
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422


def test_transition_entry_missing_field():
    tr = {"a": {"fail": "1/4"},
          "b": {"fail": "1/5", "recover": "2/3"}}
    body = make_body(BASE_EV, BASE_GATES, "top", 2, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "recover" in r.json()["detail"]


def test_steps_out_of_range():
    for bad in (0, 11):
        r = post_body(steps=bad)
        assert r.status_code == 422, (bad, r.text)


def test_steps_must_be_integer():
    r = post_body(steps=2.5)
    assert r.status_code == 422
    r = post_body(steps="2")
    assert r.status_code == 422
    # 严格整数：布尔值同样拒绝
    r = post_body(steps=True)
    assert r.status_code == 422


def test_more_than_eight_events_rejected():
    ev = {f"e{i}": "1/2" for i in range(9)}
    tr = {e: {"fail": "1/2", "recover": "1/2"} for e in ev}
    body = make_body(ev, [], "e0", 1, tr)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422


def test_transitions_field_required():
    body = make_body(BASE_EV, BASE_GATES, "top", 2, BASE_TR)
    del body["transitions"]
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "transitions" in r.json()["detail"]


def test_extra_top_level_field_rejected():
    r = post_body(unknown=1)
    assert r.status_code == 422


def test_unknown_top_reference():
    r = client.post(
        "/v1/first-failure-time",
        json=make_body(BASE_EV, BASE_GATES, "nope", 2, BASE_TR),
    )
    assert r.status_code == 422 and "顶层" in r.json()["detail"]


def test_unknown_gate_reference():
    gates = [("g", "AND", ["a", "x"])]
    body = make_body(BASE_EV, gates, "g", 2, BASE_TR)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "未知" in r.json()["detail"]


def test_cycle_rejected():
    gates = [("g", "OR", ["a", "g"])]
    body = make_body(BASE_EV, gates, "g", 2, BASE_TR)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "环" in r.json()["detail"]


def test_bad_event_probability_rejected():
    ev = {"a": "3/2", "b": "1/3"}
    body = make_body(ev, BASE_GATES, "top", 2, BASE_TR)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "概率" in r.json()["detail"]


def test_not_gate_arity_rejected():
    gates = [("g", "NOT", ["a", "b"]), ("top", "OR", ["g", "a"])]
    body = make_body(BASE_EV, gates, "top", 2, BASE_TR)
    r = client.post("/v1/first-failure-time", json=body)
    assert r.status_code == 422 and "NOT" in r.json()["detail"]
