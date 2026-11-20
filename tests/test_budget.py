import pytest

import budget
from budget import BudgetExceeded, Container


@pytest.fixture(autouse=True)
def _ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_BUDGET_LEDGER", str(tmp_path / "ledger.jsonl"))


def test_container_price():
    c = Container("L40S", cpu_cores=4, memory_gib=64)
    # 0.000542 + 4*0.0000131 + 64*0.00000222 per second
    assert c.usd_per_sec == pytest.approx(0.000542 + 0.0000524 + 0.00014208)
    assert 2.6 < c.usd_per_hour < 2.8


def test_reserve_refuses_over_cap():
    cap = budget.load_caps()["gpu"]
    budget.reserve("a", "gpu", cap - 1.0)
    with pytest.raises(BudgetExceeded):
        budget.reserve("b", "gpu", 2.0)


def test_gpu_job_settles_actual_not_worst():
    c = Container("L40S", 4, 64)
    with budget.gpu_job("quick", c, timeout_s=3600) as info:
        assert info["worst_case_usd"] == pytest.approx(c.usd_per_hour)
        assert budget.spent("gpu") == pytest.approx(c.usd_per_hour)
    assert budget.spent("gpu") < 0.01  # settled at a fraction of a second


def test_failed_job_still_settles():
    c = Container("L40S", 4, 64)
    with pytest.raises(RuntimeError):
        with budget.gpu_job("boom", c, timeout_s=600):
            raise RuntimeError("x")
    assert budget.spent("gpu") < 0.01


def test_api_and_manual():
    with budget.api_job("ledger-pilot", 5.0) as info:
        info["actual_usd"] = 1.25
    budget.record_manual("bill", "api", 0.75, "from invoice")
    assert budget.spent("api") == pytest.approx(2.0)
    assert budget.spent("gpu") == 0


def test_scaledown_and_unclamped_overrun(monkeypatch):
    c = Container("L40S", 4, 48)
    times = iter([100.0, 125.0])  # 25s wall time on a 10s timeout -> must NOT clamp to 10s
    monkeypatch.setattr(budget.time, "monotonic", lambda: next(times))
    with budget.gpu_job("overrun", c, timeout_s=10, scaledown_s=15) as info:
        assert info["worst_case_usd"] == pytest.approx(25 * c.usd_per_sec)
    assert budget.spent("gpu") == pytest.approx((25 + 15) * c.usd_per_sec, abs=1e-6)


def test_keyboard_interrupt_leaves_worst_case_reserved():
    c = Container("L40S", 4, 48)
    with pytest.raises(KeyboardInterrupt):
        with budget.gpu_job("interrupted", c, timeout_s=600, scaledown_s=15):
            raise KeyboardInterrupt
    assert budget.spent("gpu") == pytest.approx(615 * c.usd_per_sec)
