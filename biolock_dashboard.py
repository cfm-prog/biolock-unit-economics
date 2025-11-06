
# BIOLock — Streamlit Dashboard (RU)
# Три сценария + экспорт/импорт, сезонность, setup-fee в кэше, ролевая вместимость,
# NRR и точка безубыточности, Монте-Карло и торнадо-чувствительность.
# Запуск: streamlit run biolock_dashboard.py

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal, Tuple
import io
import json
# --- БАЗОВЫЕ ДАННЫЕ ДЛЯ ТЕСТА ---
tariffs = ["Academic", "Basic", "Business"]
segments = ["SMB Pharma/CRO", "Enterprise/Regulator"]
base_price = 25000       # средний ARPU, ₽/мес
base_customers = 20      # старт клиентов
base_growth = 0.15       # прирост клиентов в месяц
base_churn = 0.05        # отток клиентов
months = 36              # горизонт прогноза (мес.)
# -----------------------------
# Вспомогательные структуры
# -----------------------------
@dataclass
class Tariff:
    name: str
    price_month: float
    var_cost_month: float
    setup_fee: float = 0.0

@dataclass
class Segment:
    name: str
    tariff_mix: Dict[str, float]          # {tariff: share}
    discount_pct: float = 0.0             # скидка на прайс
    retention_mode: Literal["constant_churn","curve"] = "constant_churn"
    monthly_churn: float = 0.03
    retention_curve: Optional[List[float]] = None

@dataclass
class Channel:
    name: str
    monthly_spend: float
    cpa_paid: Optional[float] = None      # ₽ за платящего (если известен CPA)
    cpl: Optional[float] = None           # ₽ за лид (если считаем через воронку)
    funnel: Optional[Dict[str, float]] = None  # конверсии между этапами (lead->mql->sql->pilot->paid)

@dataclass
class CostStructure:
    fixed_monthly: float
    support_per_logo_month: float = 0.0

@dataclass
class Assumptions:
    horizon_months: int = 36
    discount_rate_annual: float = 0.0
    pilot_to_paid_ratio: float = 1.0
    implementation_capacity_pm: Optional[int] = None  # итоговая вместимость (если None — рассчитываем по ролям)
    # Ролевая вместимость
    impl_slots_engineering: Optional[int] = None
    impl_slots_data: Optional[int] = None
    impl_slots_success: Optional[int] = None
    req_slots_engineering_per_logo: float = 1.0
    req_slots_data_per_logo: float = 0.5
    req_slots_success_per_logo: float = 0.5
    # Сезонность (12 месяцев, среднее ~1.0)
    seasonality_12: Optional[List[float]] = None

@dataclass
class Scenario:
    name: str
    tariffs: List[Tariff]
    segments: List[Segment]
    channels: List[Channel]
    costs: CostStructure
    asm: Assumptions

# -----------------------------
# Расчётные функции
# -----------------------------
STAGES = ["lead","mql","sql","pilot","paid"]

def discount_factors(months: int, annual_rate: float) -> np.ndarray:
    if annual_rate <= 0: return np.ones(months)
    monthly = (1 + annual_rate) ** (1/12) - 1
    return 1 / (1 + monthly) ** np.arange(1, months+1)

def monthly_retention_curve(horizon: int, mode: str, monthly_churn: float, curve: Optional[List[float]]) -> np.ndarray:
    if mode == "curve" and curve:
        arr = np.array(curve, dtype=float)
        if len(arr) < horizon:
            arr = np.concatenate([arr, np.full(horizon - len(arr), arr[-1])])
        return np.clip(arr[:horizon], 0, 1)
    s = 1.0
    surv = []
    for _ in range(horizon):
        s *= (1 - monthly_churn)
        surv.append(s)
    return np.array(surv)

def compute_channel_cac(ch: Channel) -> Tuple[float, int]:
    """Возвращает (CAC, платящие/мес)."""
    if ch.cpa_paid and ch.cpa_paid > 0:
        paid = int(ch.monthly_spend / ch.cpa_paid)
        cac = ch.monthly_spend / max(paid,1)
        return cac, paid
    if ch.cpl and ch.funnel:
        leads = int(ch.monthly_spend / ch.cpl) if ch.cpl>0 else 0
        conv = 1.0
        for k in STAGES:
            if k in ch.funnel:
                conv *= float(ch.funnel[k])
        paid = int(leads * conv)
        cac = ch.monthly_spend / max(paid,1)
        return cac, paid
    return 0.0, 0

def blend_cac(channels: List[Channel]) -> Tuple[float,int,float]:
    spend, paid = 0.0, 0
    for ch in channels:
        _, p = compute_channel_cac(ch)
        spend += ch.monthly_spend
        paid += p
    cac = spend/paid if paid>0 else 0.0
    return cac, paid, spend

def effective_price(t: Tariff, seg: Segment) -> float:
    return t.price_month * (1 - seg.discount_pct)

# Юнит-метрики по каждой паре тариф×сегмент

def unit_metrics(sc: Scenario) -> pd.DataFrame:
    H = sc.asm.horizon_months
    disc_vec = discount_factors(H, sc.asm.discount_rate_annual)
    cac_blend, _, _ = blend_cac(sc.channels)
    rows = []
    for seg in sc.segments:
        S = monthly_retention_curve(H, seg.retention_mode, seg.monthly_churn, seg.retention_curve)
        for t in sc.tariffs:
            price_eff = effective_price(t, seg)
            gm = price_eff - t.var_cost_month - sc.costs.support_per_logo_month
            ltv = float((gm * S * disc_vec).sum())
            payback = None
            cum = 0.0
            for i, s in enumerate(S, start=1):
                cum += gm * s
                if cac_blend>0 and cum >= cac_blend:
                    payback = i
                    break
            ratio = ltv / cac_blend if cac_blend>0 else None
            rows.append({
                "Тариф": t.name,
                "Сегмент": seg.name,
                "GM/клиент-месяц": gm,
                "LTV (дисконт.)": ltv,
                "Payback (мес)": payback,
                "LTV/CAC": ratio
            })
    return pd.DataFrame(rows)

# Эффективная вместимость из ролей

def capacity_from_roles(asm: Assumptions) -> Optional[int]:
    if asm.implementation_capacity_pm and asm.implementation_capacity_pm>0:
        return int(asm.implementation_capacity_pm)
    # если итоговая вместимость не задана — считаем как минимум по ролям
    role_caps = []
    if asm.impl_slots_engineering is not None:
        role_caps.append(int(asm.impl_slots_engineering // max(asm.req_slots_engineering_per_logo, 1e-6)))
    if asm.impl_slots_data is not None:
        role_caps.append(int(asm.impl_slots_data // max(asm.req_slots_data_per_logo, 1e-6)))
    if asm.impl_slots_success is not None:
        role_caps.append(int(asm.impl_slots_success // max(asm.req_slots_success_per_logo, 1e-6)))
    return int(min(role_caps)) if role_caps else None

# Помесячная динамика логотипов и финансов (учёт setup-fee и сезонности)

def run_forecast(sc: Scenario) -> pd.DataFrame:
    H = sc.asm.horizon_months
    active = {(seg.name, t.name): 0.0 for seg in sc.segments for t in sc.tariffs}
    rows = []
    seas = sc.asm.seasonality_12 or [1.0]*12
    cap = capacity_from_roles(sc.asm)

    prev_revenue = 0.0
    for m in range(1, H+1):
        # Новые клиенты из каналов с сезонностью
        total_new_raw = 0
        ad_spend = 0.0
        for ch in sc.channels:
            _, paid = compute_channel_cac(ch)
            total_new_raw += paid
            ad_spend += ch.monthly_spend
        total_new = int(round(total_new_raw * sc.asm.pilot_to_paid_ratio * seas[(m-1)%12]))
        if cap is not None:
            total_new = min(total_new, cap)

        # распределяем по сегментам поровну (упрощение) и по тариф-миксу в сегменте
        seg_count = len(sc.segments)
        alloc_seg = {seg.name: total_new//seg_count for seg in sc.segments}
        for i in range(total_new % seg_count):
            alloc_seg[sc.segments[i].name] += 1

        # churn существующих
        for seg in sc.segments:
            churn = seg.monthly_churn if seg.retention_mode=="constant_churn" else (1 - (seg.retention_curve[0] if seg.retention_curve else 0.97))
            for t in sc.tariffs:
                a = active[(seg.name, t.name)]
                active[(seg.name, t.name)] = max(a * (1 - churn), 0)

        # добавляем новые и считаем setup-fee
        setup_cash = 0.0
        new_rev_this_month = 0.0
        for seg in sc.segments:
            for t in sc.tariffs:
                share = seg.tariff_mix.get(t.name, 0.0)
                add = alloc_seg[seg.name] * share
                active[(seg.name, t.name)] += add
                # setup-fee
                setup_cash += add * t.setup_fee
                new_rev_this_month += add * effective_price(t, seg)

        # финансы
        revenue = 0.0
        var_costs = 0.0
        support_costs = 0.0
        for seg in sc.segments:
            for t in sc.tariffs:
                a = active[(seg.name, t.name)]
                price_eff = effective_price(t, seg)
                revenue += a * price_eff
                var_costs += a * t.var_cost_month
                support_costs += a * sc.costs.support_per_logo_month
        gross_margin = revenue - var_costs - support_costs
        contribution = gross_margin - ad_spend
        ebitda = contribution - sc.costs.fixed_monthly + setup_cash  # setup-fee учитываем как приток в месяц сделки

        # NRR (примерная оценка): существующий MRR после churn / MRR предыдущего месяца
        existing_prev = max(prev_revenue, 1e-9)
        nrr = (revenue - new_rev_this_month) / existing_prev if prev_revenue>0 else None
        prev_revenue = revenue

        rows.append({
            "Месяц": m,
            "Активные клиенты": sum(active.values()),
            "Выручка": revenue,
            "Выручка (новые)": new_rev_this_month,
            "Валовая маржа": gross_margin,
            "Маркетинг": ad_spend,
            "Setup-fee cash": setup_cash,
            "EBITDA": ebitda,
            "NRR": nrr
        })
    return pd.DataFrame(rows)

# Монте-Карло (небольшое N для скорости)

def monte_carlo(sc: Scenario, runs: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(runs):
        sc2 = Scenario(
            name=sc.name,
            tariffs=[Tariff(**asdict(t)) for t in sc.tariffs],
            segments=[Segment(**asdict(s)) for s in sc.segments],
            channels=[Channel(**asdict(c)) for c in sc.channels],
            costs=CostStructure(**asdict(sc.costs)),
            asm=Assumptions(**asdict(sc.asm))
        )
        # шумим churn, CPA/CPL, spend, скидки, цены
        for s in sc2.segments:
            s.monthly_churn = float(np.clip(rng.normal(s.monthly_churn, 0.005), 0.003, 0.25))
            s.discount_pct = float(np.clip(rng.normal(s.discount_pct, 0.02), 0.0, 0.5))
        for t in sc2.tariffs:
            t.price_month = float(max(0.0, rng.normal(t.price_month, 0.1*t.price_month)))
        for c in sc2.channels:
            c.monthly_spend = float(max(0.0, rng.normal(c.monthly_spend, 0.1*c.monthly_spend)))
            if c.cpa_paid:
                c.cpa_paid = float(max(1e3, rng.normal(c.cpa_paid, 0.15*c.cpa_paid)))
            if c.cpl:
                c.cpl = float(max(1e3, rng.normal(c.cpl, 0.15*c.cpl)))
        df = run_forecast(sc2)
        results.append({
            "Суммарная выручка": float(df["Выручка"].sum()),
            "Суммарная EBITDA": float(df["EBITDA"].sum()),
            "MRR (последний месяц)": float(df.iloc[-1]["Выручка"]),
            "Месяцев с прибылью": int((df["EBITDA"]>0).sum()),
            "Первый прибыльный месяц": int([df["EBITDA"]>0][0])+1 if (df["EBITDA"]>0).any() else None
        })
    return pd.DataFrame(results)

# Торнадо-чувствительность (влияние на суммарную EBITDA)

def tornado_sensitivity(sc: Scenario, base_df: pd.DataFrame, deltas: Dict[str, float]) -> pd.DataFrame:
    base_ebitda = float(base_df["EBITDA"].sum())
    rows = []
    def _apply_and_sum(copy_fn):
        sc2 = copy_fn()
        df = run_forecast(sc2)
        return float(df["EBITDA"].sum())
    # параметры: price, churn, CPA, fixed
    # price +/−
    for name, mult in [("Цена +", 1+deltas['price']), ("Цена −", 1-deltas['price'])]:
        def cpy():
            sc2 = Scenario(
                name=sc.name,
                tariffs=[Tariff(t.name, t.price_month*mult, t.var_cost_month, t.setup_fee) for t in sc.tariffs],
                segments=[Segment(**asdict(s)) for s in sc.segments],
                channels=[Channel(**asdict(c)) for c in sc.channels],
                costs=CostStructure(**asdict(sc.costs)),
                asm=Assumptions(**asdict(sc.asm))
            )
            return sc2
        eb = _apply_and_sum(cpy)
        rows.append({"Параметр":"Цена","Вариант":name,"Δ EBITDA": eb - base_ebitda})
    # churn +/−
    for name, mult in [("Churn +", 1+deltas['churn']), ("Churn −", 1-deltas['churn'])]:
        def cpy():
            sc2 = Scenario(
                name=sc.name,
                tariffs=[Tariff(**asdict(t)) for t in sc.tariffs],
                segments=[Segment(s.name, dict(s.tariff_mix), s.discount_pct, s.retention_mode, s.monthly_churn*mult, s.retention_curve) for s in sc.segments],
                channels=[Channel(**asdict(c)) for c in sc.channels],
                costs=CostStructure(**asdict(sc.costs)),
                asm=Assumptions(**asdict(sc.asm))
            )
            return sc2
        eb = _apply_and_sum(cpy)
        rows.append({"Параметр":"Churn","Вариант":name,"Δ EBITDA": eb - base_ebitda})
    # CPA +/− (и CPL)
    for name, mult in [("CAC +", 1+deltas['cac']), ("CAC −", 1-deltas['cac'])]:
        def cpy():
            sc2 = Scenario(
                name=sc.name,
                tariffs=[Tariff(**asdict(t)) for t in sc.tariffs],
                segments=[Segment(**asdict(s)) for s in sc.segments],
                channels=[Channel(c.name, c.monthly_spend, cpa_paid=c.cpa_paid*mult if c.cpa_paid else None,
                                  cpl=c.cpl*mult if c.cpl else None, funnel=(dict(c.funnel) if c.funnel else None)) for c in sc.channels],
                costs=CostStructure(**asdict(sc.costs)),
                asm=Assumptions(**asdict(sc.asm))
            )
            return sc2
        eb = _apply_and_sum(cpy)
        rows.append({"Параметр":"CAC","Вариант":name,"Δ EBITDA": eb - base_ebitda})
    # fixed +/−
    for name, mult in [("Fixed +", 1+deltas['fixed']), ("Fixed −", 1-deltas['fixed'])]:
        def cpy():
            sc2 = Scenario(
                name=sc.name,
                tariffs=[Tariff(**asdict(t)) for t in sc.tariffs],
                segments=[Segment(**asdict(s)) for s in sc.segments],
                channels=[Channel(**asdict(c)) for c in sc.channels],
                costs=CostStructure(fixed_monthly=sc.costs.fixed_monthly*mult, support_per_logo_month=sc.costs.support_per_logo_month),
                asm=Assumptions(**asdict(sc.asm))
            )
            return sc2
        eb = _apply_and_sum(cpy)
        rows.append({"Параметр":"Fixed","Вариант":name,"Δ EBITDA": eb - base_ebitda})
    df = pd.DataFrame(rows)
    # для торнадо возьмем максимальные отклонения по модулю для каждого параметра
    agg = df.groupby("Параметр")["Δ EBITDA"].apply(lambda x: x.loc[x.abs().idxmax()])
    return agg.sort_values(key=lambda s: s.abs(), ascending=True).reset_index()

# -----------------------------
# UI — Streamlit
# -----------------------------
st.set_page_config(page_title="BIOLock Unit Economics (RU)", layout="wide")
st.title("BIOLock — Юнит-экономика (Россия)")
st.caption("Интерактивная модель: тарифы, сегменты, каналы, издержки, сценарии, сезонность, NRR и Монте-Карло")

# --- Боковая панель: горизонт, издержки, capacity ---
st.sidebar.header("⚙️ Общие настройки")
horizon = st.sidebar.slider("Горизонт, мес", 12, 60, 36, 1)
fixed_monthly = st.sidebar.number_input("Фиксированные издержки (₽/мес)", 0, 20_000_000, 833_000, 10_000)
support_month = st.sidebar.number_input("Поддержка на клиента (₽/мес)", 0, 200_000, 0, 1_000)

st.sidebar.subheader("👥 Вместимость внедрений")
cap_mode = st.sidebar.radio("Как считать вместимость?", ["Итоговая цифра", "По ролям"], index=1)
if cap_mode == "Итоговая цифра":
    capacity = st.sidebar.number_input("Лимит внедрений в месяц (шт)", 0, 200, 6, 1)
    eng_slots = data_slots = cs_slots = None
else:
    capacity = 0
    eng_slots = st.sidebar.number_input("Слотов инженерии/мес", 0, 200, 6, 1)
    data_slots = st.sidebar.number_input("Слотов data/мес", 0, 200, 4, 1)
    cs_slots = st.sidebar.number_input("Слотов customer success/мес", 0, 200, 4, 1)
    st.sidebar.caption("Требования на 1 новый логотип, слотов:")
    req_e = st.sidebar.number_input("Треб. инженерии", 0.0, 10.0, 1.0, 0.1)
    req_d = st.sidebar.number_input("Треб. data", 0.0, 10.0, 0.5, 0.1)
    req_s = st.sidebar.number_input("Треб. success", 0.0, 10.0, 0.5, 0.1)

st.sidebar.subheader("📅 Сезонность привлечения")
season_profile = st.sidebar.selectbox("Профиль", ["Нет", "Лёгкая", "Сильная"]) 
if season_profile == "Нет":
    seasonality = [1.0]*12
elif season_profile == "Лёгкая":
    seasonality = [0.9,0.95,0.95,1.0,1.05,1.1,1.1,1.05,1.0,0.95,0.9,0.85]
else:
    seasonality = [0.8,0.9,0.95,1.0,1.1,1.2,1.25,1.15,1.0,0.9,0.85,0.8]

pilot_to_paid = st.sidebar.slider("Доля пилотов → платящие", 0.0, 1.0, 0.8, 0.05)
disc_annual = st.sidebar.slider("Дисконт (годовой) для LTV", 0.0, 0.3, 0.0, 0.01)

# --- TABs ---
t1, t2, t3, t4, t5, t6 = st.tabs(["💰 Тарифы и цены","📊 Сегменты и удержание","📣 Каналы маркетинга","🏗 Экономика","📈 Результаты и графики","📦 Экспорт/Импорт сценария"]) 

# ===== Тарифы =====
with t1:
    st.subheader("Тарифы")
    cols = st.columns(3)
    tariffs: List[Tariff] = []
    # дефолты
    defaults = [
        ("Academic", 20_000, 8_000, 0),
        ("Basic", 50_000, 8_000, 0),
        ("Business", 100_000, 8_000, 50_000),
    ]
    for i, (name, p, vc, setup) in enumerate(defaults):
        with cols[i]:
            st.markdown(f"**{name}**")
            price = st.number_input(f"Цена/мес — {name}", 0, 5_000_000, p, 1_000, key=f"price_{name}")
            varc = st.number_input(f"Перем. издержки/мес — {name}", 0, 5_000_000, vc, 1_000, key=f"var_{name}")
            setup_fee = st.number_input(f"Setup fee (разово) — {name}", 0, 5_000_000, setup, 10_000, key=f"setup_{name}")
            tariffs.append(Tariff(name=name, price_month=price, var_cost_month=varc, setup_fee=setup_fee))
    st.info("Setup fee учитывается в денежном потоке месяца подключения (не в MRR).")

# ===== Сегменты =====
with t2:
    st.subheader("Сегменты и удержание")
    segs: List[Segment] = []
    seg_cols = st.columns(2)
    # SMB
    with seg_cols[0]:
        st.markdown("**SMB Pharma/CRO**")
        disc = st.slider("Скидка (SMB)", 0.0, 0.5, 0.05, 0.01)
        churn = st.slider("Churn/мес (SMB)", 0.0, 0.2, 0.03, 0.005)
        st.caption("Доли тарифов (сумма≈1)")
        m1 = st.number_input("Academic (SMB)", 0.0, 1.0, 0.20, 0.05)
        m2 = st.number_input("Basic (SMB)", 0.0, 1.0, 0.60, 0.05)
        m3 = st.number_input("Business (SMB)", 0.0, 1.0, 0.20, 0.05)
        s = m1+m2+m3
        if s==0: s=1.0
        mix_smb = {"Academic": m1/s, "Basic": m2/s, "Business": m3/s}
        segs.append(Segment(name="SMB Pharma/CRO", tariff_mix=mix_smb, discount_pct=disc, monthly_churn=churn))
    # Enterprise
    with seg_cols[1]:
        st.markdown("**Enterprise/Regulator**")
        disc = st.slider("Скидка (Enterprise)", 0.0, 0.5, 0.10, 0.01)
        churn = st.slider("Churn/мес (Enterprise)", 0.0, 0.2, 0.02, 0.005)
        st.caption("Доли тарифов (сумма≈1)")
        m1 = st.number_input("Academic (Ent)", 0.0, 1.0, 0.10, 0.05)
        m2 = st.number_input("Basic (Ent)", 0.0, 1.0, 0.40, 0.05)
        m3 = st.number_input("Business (Ent)", 0.0, 1.0, 0.50, 0.05)
        s = m1+m2+m3
        if s==0: s=1.0
        mix_ent = {"Academic": m1/s, "Basic": m2/s, "Business": m3/s}
        segs.append(Segment(name="Enterprise/Regulator", tariff_mix=mix_ent, discount_pct=disc, monthly_churn=churn))

# ===== Каналы =====
with t3:
    st.subheader("Каналы маркетинга")
    ch_cols = st.columns(3)
    channels: List[Channel] = []
    with ch_cols[0]:
        st.markdown("**Events**")
        spend = st.number_input("Бюджет/мес (Events)", 0, 20_000_000, 400_000, 10_000)
        cpa = st.number_input("CPA (₽/платящий)", 1000, 5_000_000, 700_000, 10_000)
        channels.append(Channel(name="Events", monthly_spend=spend, cpa_paid=cpa))
    with ch_cols[1]:
        st.markdown("**Outbound SDR (воронка)**")
        spend = st.number_input("Бюджет/мес (SDR)", 0, 20_000_000, 300_000, 10_000)
        cpl = st.number_input("CPL (₽/лид)", 1000, 5_000_000, 10_000, 1000)
        col2 = st.container()
        with col2:
            st.caption("Конверсии (0..1)")
            c_lead = st.number_input("lead→mql", 0.0, 1.0, 0.50, 0.05)
            c_mql = st.number_input("mql→sql", 0.0, 1.0, 0.40, 0.05)
            c_sql = st.number_input("sql→pilot", 0.0, 1.0, 0.50, 0.05)
            c_pilot = st.number_input("pilot→paid", 0.0, 1.0, 0.25, 0.05)
        channels.append(Channel(name="Outbound SDR", monthly_spend=spend, cpl=cpl,
                                funnel={"lead":1.0, "mql":c_lead, "sql":c_mql, "pilot":c_sql, "paid":c_pilot}))
    with ch_cols[2]:
        st.markdown("**Partners**")
        spend = st.number_input("Бюджет/мес (Partners)", 0, 20_000_000, 200_000, 10_000)
        cpa = st.number_input("CPA (₽/платящий)", 1000, 5_000_000, 500_000, 10_000)
        channels.append(Channel(name="Partners", monthly_spend=spend, cpa_paid=cpa))

# ===== Экономика =====
with t4:
    st.subheader("Экономические параметры")
    st.write("Вы настраивали общие издержки и сезонность в сайдбаре. Ниже — параметры LTV/дисконт." )

# Сценарные множители — в сайдбаре, чтобы было видно всегда
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Сценарные множители")
mult_price_opt = st.sidebar.slider("Цена: Optimistic ×", 0.6, 1.6, 1.10, 0.01)
mult_churn_opt = st.sidebar.slider("Churn: Optimistic ×", 0.4, 1.2, 0.85, 0.01)
mult_cac_opt = st.sidebar.slider("CPA/CPL: Optimistic ×", 0.6, 1.6, 0.90, 0.01)
mult_price_pes = st.sidebar.slider("Цена: Pessimistic ×", 0.6, 1.6, 0.90, 0.01)
mult_churn_pes = st.sidebar.slider("Churn: Pessimistic ×", 0.8, 2.0, 1.15, 0.01)
mult_cac_pes = st.sidebar.slider("CPA/CPL: Pessимistic ×", 0.8, 2.0, 1.15, 0.01)

# Сборка базового сценария
asm = Assumptions(
    horizon_months=horizon,
    discount_rate_annual=disc_annual,
    pilot_to_paid_ratio=pilot_to_paid,
    implementation_capacity_pm=int(capacity) if cap_mode=="Итоговая цифра" and capacity>0 else None,
    impl_slots_engineering=eng_slots if cap_mode=="По ролям" else None,
    impl_slots_data=data_slots if cap_mode=="По ролям" else None,
    impl_slots_success=cs_slots if cap_mode=="По ролям" else None,
    req_slots_engineering_per_logo=req_e if cap_mode=="По ролям" else 1.0,
    req_slots_data_per_logo=req_d if cap_mode=="По ролям" else 0.5,
    req_slots_success_per_logo=req_s if cap_mode=="По ролям" else 0.5,
    seasonality_12=seasonality
)

base = Scenario(
    name="Base",
    tariffs=tariffs,
    segments=segs,
    channels=channels,
    costs=CostStructure(fixed_monthly=fixed_monthly, support_per_logo_month=support_month),
    asm=asm
)

# Функция для создания варианта сценария с множителями

def scenario_with_multipliers(sc: Scenario, name: str, m_price: float, m_churn: float, m_cac: float) -> Scenario:
    sc2 = Scenario(
        name=name,
        tariffs=[Tariff(t.name, t.price_month*m_price, t.var_cost_month, t.setup_fee) for t in sc.tariffs],
        segments=[Segment(s.name, dict(s.tariff_mix), min(max(s.discount_pct,0),0.5), s.retention_mode, s.monthly_churn*m_churn, s.retention_curve) for s in sc.segments],
        channels=[],
        costs=CostStructure(sc.costs.fixed_monthly, sc.costs.support_per_logo_month),
        asm=Assumptions(**asdict(sc.asm))
    )
    for c in sc.channels:
        ch = Channel(c.name, c.monthly_spend, cpa_paid=c.cpa_paid*m_cac if c.cpa_paid else None,
                     cpl=c.cpl*m_cac if c.cpl else None, funnel=(dict(c.funnel) if c.funnel else None))
        sc2.channels.append(ch)
    return sc2

opt = scenario_with_multipliers(base, "Optimistic", mult_price_opt, mult_churn_opt, mult_cac_opt)
pes = scenario_with_multipliers(base, "Pessimistic", mult_price_pes, mult_churn_pes, mult_cac_pes)

# ===== Результаты =====
with t5:
    st.subheader("Итоговые метрики и графики")
    # Unit metrics
    st.markdown("### Юнит-метрики (по тарифам × сегментам)")
    um_base = unit_metrics(base)
    um_opt = unit_metrics(opt)
    um_pes = unit_metrics(pes)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write("**Base**")
        st.dataframe(um_base)
    with c2:
        st.write("**Optimistic**")
        st.dataframe(um_opt)
    with c3:
        st.write("**Pessimistic**")
        st.dataframe(um_pes)

    # Forecast lines
    st.markdown("### Помесячный прогноз")
    df_b = run_forecast(base)
    df_o = run_forecast(opt)
    df_p = run_forecast(pes)

    # Break-even month (пересечение нуля)
    def break_even_month(df):
        idx = np.where((df["EBITDA"].values>0))[0]
        return int(idx[0]+1) if len(idx)>0 else None
    be_b, be_o, be_p = break_even_month(df_b), break_even_month(df_o), break_even_month(df_p)

    # График MRR (выручка)
    fig_mrr = go.Figure()
    fig_mrr.add_trace(go.Scatter(x=df_b["Месяц"], y=df_b["Выручка"], name=f"MRR — Base (BE={be_b})"))
    fig_mrr.add_trace(go.Scatter(x=df_o["Месяц"], y=df_o["Выручка"], name=f"MRR — Optimistic (BE={be_o})"))
    fig_mrr.add_trace(go.Scatter(x=df_p["Месяц"], y=df_p["Выручка"], name=f"MRR — Pessimistic (BE={be_p})"))
    fig_mrr.update_layout(title="MRR (выручка по месяцам)", xaxis_title="Месяц", yaxis_title="₽/мес")
    st.plotly_chart(fig_mrr, use_container_width=True)

    # График EBITDA
    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=df_b["Месяц"], y=df_b["EBITDA"], name=f"EBITDA — Base (BE={be_b})"))
    fig_e.add_trace(go.Scatter(x=df_o["Месяц"], y=df_o["EBITDA"], name=f"EBITDA — Optimistic (BE={be_o})"))
    fig_e.add_trace(go.Scatter(x=df_p["Месяц"], y=df_p["EBITDA"], name=f"EBITDA — Pessimistic (BE={be_p})"))
    for x in [be_b, be_o, be_p]:
        if x:
            fig_e.add_vline(x=x, line_dash="dash", line_color="#888")
    fig_e.update_layout(title="EBITDA по месяцам (вертикали — break-even)", xaxis_title="Месяц", yaxis_title="₽/мес")
    st.plotly_chart(fig_e, use_container_width=True)

    # График Активные клиенты
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_b["Месяц"], y=df_b["Активные клиенты"], name="Клиенты — Base"))
    fig_c.add_trace(go.Scatter(x=df_o["Месяц"], y=df_o["Активные клиенты"], name="Клиенты — Optimistic"))
    fig_c.add_trace(go.Scatter(x=df_p["Месяц"], y=df_p["Активные клиенты"], name="Клиенты — Pessimistic"))
    fig_c.update_layout(title="Активные клиенты", xaxis_title="Месяц", yaxis_title="Кол-во")
    st.plotly_chart(fig_c, use_container_width=True)

    # NRR (Base)
    fig_nrr = go.Figure()
    fig_nrr.add_trace(go.Scatter(x=df_b["Месяц"], y=df_b["NRR"], name="NRR — Base"))
    fig_nrr.update_layout(title="NRR по месяцам (оценка)", xaxis_title="Месяц", yaxis_title="NRR")
    st.plotly_chart(fig_nrr, use_container_width=True)

    # Монте-Карло
    st.markdown("### Монте-Карло (по Base)")
    n_runs = st.slider("Число прогонов", 50, 1000, 200, 50)
    if st.button("Запустить Монте-Карло"):
        dist = monte_carlo(base, runs=n_runs)
        colA, colB = st.columns(2)
        with colA:
            fig_hist = px.histogram(dist, x="Суммарная EBITDA", nbins=30, title="Распределение: суммарная EBITDA")
            st.plotly_chart(fig_hist, use_container_width=True)
        with colB:
            q = dist.quantile([0.1, 0.5, 0.9])
            st.write("Квантили (0.1 / 0.5 / 0.9):")
            st.dataframe(q)

    # Торнадо-чувствительность
    st.markdown("### Чувствительность (торнадо)")
    deltas = {
        'price': st.slider("Δ Цена, %", 0.01, 0.50, 0.20, 0.01),
        'churn': st.slider("Δ Churn, %", 0.01, 0.50, 0.20, 0.01),
        'cac': st.slider("Δ CAC, %", 0.01, 0.50, 0.20, 0.01),
        'fixed': st.slider("Δ Fixed, %", 0.01, 0.50, 0.20, 0.01),
    }
    if st.button("Построить торнадо"):
        # конвертируем проценты в множители
        d = {k: float(v) for k,v in deltas.items()}
        df_t = tornado_sensitivity(base, df_b, d)
        fig_t = go.Figure(go.Bar(x=df_t["Δ EBITDA"], y=df_t["Параметр"], orientation='h'))
        fig_t.update_layout(title="Вклад параметров в отклонение EBITDA (торнадо)", xaxis_title="Δ EBITDA (₽)", yaxis_title="Параметр")
        st.plotly_chart(fig_t, use_container_width=True)

# ===== Экспорт / импорт =====
with t6:
    st.subheader("Экспорт / Импорт сценария")
    if st.button("Экспортировать текущий сценарий в JSON"):
        payload = {
            "tariffs": [asdict(t) for t in tariffs],
            "segments": [asdict(s) for s in segs],
            "channels": [asdict(c) for c in channels],
            "costs": asdict(CostStructure(fixed_monthly=fixed_monthly, support_per_logo_month=support_month)),
            "assumptions": asdict(asm)
        }
        b = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
        st.download_button("Скачать JSON", data=b, file_name="biolock_scenario.json", mime="application/json")

    st.markdown("---")
    uploaded = st.file_uploader("Загрузить JSON сценария", type=["json"])
    if uploaded is not None:
        data = json.load(uploaded)
        st.write("Файл прочитан. Для полной интеграции — подставьте значения в UI слева/вкладках (безопаснее).")
        st.json(data)

# Футер
st.markdown("---")
st.caption("BIOLock Unit Economics — интерактивная модель. Изменяйте параметры сверху и наблюдайте, как меняются метрики. © 2025")
