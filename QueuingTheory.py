import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(page_title="Markovian Queueing Systems", layout="wide")


# ── Constants ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"

COLORS = [
    "#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
    "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"
]


# ── Styles ───────────────────────────────────────────────────────────────────
watermark_html = f"""
<style>
.watermark {{
    position: fixed;
    top: 150px;
    right: 25px;
    opacity: 0.95;
    font-size: 22px;
    font-weight: 900;
    color: #ff4b4b;
    text-shadow: 1px 1px 2px #000;
    z-index: 9999;
    pointer-events: none;
}}

.info-box {{
    background-color: rgba(49, 51, 63, 0.08);
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(120, 120, 120, 0.25);
    margin-bottom: 18px;
}}

.metric-card {{
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    min-height: 142px;
}}

.metric-card h4 {{
    color: #e2e8f0;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}}

.metric-card .metric-value {{
    color: #38bdf8;
    font-size: 28px;
    font-weight: 900;
    font-family: monospace;
}}

.metric-card .metric-label {{
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
    line-height: 1.35;
}}
</style>

<div class="watermark">{WATERMARK_TEXT}</div>
"""

st.markdown(watermark_html, unsafe_allow_html=True)


# ── State-dependent server policy ────────────────────────────────────────────
def validate_thresholds(thresholds: list[int], servers: int):
    if len(thresholds) != servers:
        raise ValueError("There must be exactly one activation threshold for each server.")
    if thresholds[0] != 1:
        raise ValueError("Server 1 must be activated from state n = 1.")
    if any(t < 1 for t in thresholds):
        raise ValueError("Activation thresholds must be positive integers.")
    if any(thresholds[i] <= thresholds[i - 1] for i in range(1, len(thresholds))):
        raise ValueError("Activation thresholds must be strictly increasing.")


def operating_servers(n: int, thresholds: list[int]) -> int:
    """Number of servers switched on when there are n customers in the system."""
    if n <= 0:
        return 0
    return sum(1 for threshold in thresholds if n >= threshold)


def busy_servers(n: int, thresholds: list[int]) -> int:
    """Number of servers actually busy in state n."""
    if n <= 0:
        return 0
    return min(n, operating_servers(n, thresholds))


def total_service_rate(n: int, thresholds: list[int], mu: float) -> float:
    """Total death/service rate μ_n in state n."""
    return busy_servers(n, thresholds) * mu


def birth_rate(n: int, lam: float, finite: bool, K: int | None) -> float:
    if finite and K is not None and n >= K:
        return 0.0
    return lam


def saturation_state(thresholds: list[int], servers: int) -> int:
    """First state from which all s servers are busy and μ_n remains sμ."""
    return max(max(thresholds), servers)


# ── Stationary distributions ─────────────────────────────────────────────────
def finite_stationary_distribution(
    lam: float,
    mu: float,
    thresholds: list[int],
    K: int,
) -> np.ndarray:
    weights = np.ones(K + 1, dtype=float)
    for n in range(1, K + 1):
        lam_prev = birth_rate(n - 1, lam, True, K)
        mu_n = total_service_rate(n, thresholds, mu)
        if mu_n <= 0:
            raise ValueError(
                f"The total service rate is zero in state n = {n}. Check the server activation policy."
            )
        weights[n] = weights[n - 1] * lam_prev / mu_n

    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Could not normalize the stationary distribution.")
    return weights / total


def infinite_stationary_distribution(
    lam: float,
    mu: float,
    servers: int,
    thresholds: list[int],
    max_state: int,
) -> tuple[np.ndarray, float, float, int]:
    """
    Exact stationary probabilities for a birth-death queue with a state-dependent
    server activation policy. Once all s servers are busy, the tail is geometric.
    """
    R = saturation_state(thresholds, servers)
    tail_ratio = lam / (servers * mu)

    if tail_ratio >= 1.0:
        raise ValueError(
            "The infinite-capacity system is unstable because λ ≥ sμ after all servers are active."
        )

    # Unnormalized weights through the first state of the geometric tail.
    weights = np.ones(R + 1, dtype=float)
    for n in range(1, R + 1):
        mu_n = total_service_rate(n, thresholds, mu)
        if mu_n <= 0:
            raise ValueError(
                f"The total service rate is zero in state n = {n}. Check the server activation policy."
            )
        weights[n] = weights[n - 1] * lam / mu_n

    normalizer = weights[:R].sum() + weights[R] / (1.0 - tail_ratio)
    p0 = 1.0 / normalizer

    probs = np.zeros(max_state + 1, dtype=float)
    for n in range(max_state + 1):
        if n <= R:
            if n <= R:
                # Recurrence gives exact weight for states up to R.
                if n < len(weights):
                    probs[n] = p0 * weights[n]
        else:
            probs[n] = p0 * weights[R] * (tail_ratio ** (n - R))

    # In case max_state < R, compute missing displayed probabilities directly.
    if max_state < R:
        local_weights = np.ones(max_state + 1, dtype=float)
        for n in range(1, max_state + 1):
            local_weights[n] = local_weights[n - 1] * lam / total_service_rate(n, thresholds, mu)
        probs = p0 * local_weights

    tail = max(0.0, 1.0 - float(probs.sum()))
    return probs, tail, p0, R


# ── Queue metrics ────────────────────────────────────────────────────────────
def finite_metrics(
    lam: float,
    mu: float,
    servers: int,
    thresholds: list[int],
    K: int,
    p: np.ndarray,
) -> dict:
    states = np.arange(K + 1)
    p_block = float(p[K])
    lambda_eff = lam * (1.0 - p_block)

    busy = np.array([busy_servers(int(n), thresholds) for n in states], dtype=float)
    waiting = states - busy

    L = float(np.sum(states * p))
    Lq = float(np.sum(waiting * p))
    mean_busy = float(np.sum(busy * p))
    utilization = mean_busy / servers

    W = L / lambda_eff if lambda_eff > 1e-15 else np.nan
    Wq = Lq / lambda_eff if lambda_eff > 1e-15 else np.nan

    # PASTA: an admitted arrival waits if, after its arrival and policy reaction,
    # the number of customers exceeds the number of operating servers.
    wait_mass = 0.0
    for n in range(K):
        post_arrival = n + 1
        if post_arrival > operating_servers(post_arrival, thresholds):
            wait_mass += p[n]
    admitted_prob = 1.0 - p_block
    p_wait = wait_mass / admitted_prob if admitted_prob > 1e-15 else np.nan

    return {
        "stable": True,
        "rho": lam / (servers * mu),
        "p0": float(p[0]),
        "p_wait": p_wait,
        "blocking_probability": p_block,
        "lambda_effective": lambda_eff,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": mean_busy,
        "server_utilization": utilization,
    }


def infinite_metrics(
    lam: float,
    mu: float,
    servers: int,
    thresholds: list[int],
    p0: float,
) -> dict:
    R = saturation_state(thresholds, servers)
    r = lam / (servers * mu)
    if r >= 1.0:
        return {"stable": False, "rho": r}

    # Exact probabilities from 0 through R.
    weights = np.ones(R + 1, dtype=float)
    for n in range(1, R + 1):
        weights[n] = weights[n - 1] * lam / total_service_rate(n, thresholds, mu)
    p = p0 * weights
    pR = float(p[R])

    # Exact infinite sums: states 0,...,R-1 plus the geometric tail R,R+1,...
    L_prefix = sum(n * p[n] for n in range(R))
    L_tail = pR * (R / (1.0 - r) + r / ((1.0 - r) ** 2))
    L = float(L_prefix + L_tail)

    Lq_prefix = 0.0
    busy_prefix = 0.0
    for n in range(R):
        b = busy_servers(n, thresholds)
        Lq_prefix += (n - b) * p[n]
        busy_prefix += b * p[n]

    # For n >= R, all s servers are busy.
    Lq_tail = pR * ((R - servers) / (1.0 - r) + r / ((1.0 - r) ** 2))
    busy_tail = servers * pR / (1.0 - r)

    Lq = float(Lq_prefix + Lq_tail)
    mean_busy = float(busy_prefix + busy_tail)

    W = L / lam if lam > 0 else np.nan
    Wq = Lq / lam if lam > 0 else np.nan

    # PASTA probability that an arrival will have to wait after the activation
    # policy reacts to the new population level.
    wait_prob = 0.0
    for n in range(R):
        if (n + 1) > operating_servers(n + 1, thresholds):
            wait_prob += p[n]
    # For every pre-arrival state n >= R, all servers are already busy.
    wait_prob += pR / (1.0 - r)

    return {
        "stable": True,
        "rho": r,
        "p0": p0,
        "p_wait": float(wait_prob),
        "blocking_probability": 0.0,
        "lambda_effective": lam,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": mean_busy,
        "server_utilization": mean_busy / servers,
    }


# ── Generator and tables ─────────────────────────────────────────────────────
def build_generator_matrix(
    lam: float,
    mu: float,
    thresholds: list[int],
    max_state: int,
    finite: bool,
    K: int | None,
) -> np.ndarray:
    n_states = max_state + 1
    Q = np.zeros((n_states, n_states), dtype=float)

    for n in range(n_states):
        lam_n = birth_rate(n, lam, finite, K)
        mu_n = total_service_rate(n, thresholds, mu)

        if n + 1 < n_states and lam_n > 0:
            Q[n, n + 1] = lam_n
        if n - 1 >= 0 and mu_n > 0:
            Q[n, n - 1] = mu_n

        Q[n, n] = -(lam_n + mu_n)

    return Q


def build_rate_table(
    lam: float,
    mu: float,
    thresholds: list[int],
    max_state: int,
    finite: bool,
    K: int | None,
) -> pd.DataFrame:
    rows = []
    for n in range(max_state + 1):
        op = operating_servers(n, thresholds)
        busy = busy_servers(n, thresholds)
        waiting = max(n - busy, 0)
        rows.append({
            "State n": n,
            "Servers in operation": op,
            "Busy servers": busy,
            "Customers waiting": waiting,
            "Birth rate λₙ": birth_rate(n, lam, finite, K),
            "Death rate μₙ": total_service_rate(n, thresholds, mu),
        })
    return pd.DataFrame(rows)


def build_policy_table(thresholds: list[int], finite: bool, K: int | None) -> pd.DataFrame:
    rows = []
    for j, start in enumerate(thresholds, start=1):
        if j < len(thresholds):
            end = thresholds[j] - 1
            state_range = f"{start} to {end}" if start != end else str(start)
        else:
            if finite and K is not None:
                state_range = f"{start} to {K}"
            else:
                state_range = f"{start} and above"
        rows.append({
            "Number of customers in the system": state_range,
            "Servers in operation": j,
        })
    return pd.DataFrame(rows)


def build_birth_death_figure(
    lam: float,
    mu: float,
    thresholds: list[int],
    max_state: int,
    finite: bool,
    K: int | None,
) -> go.Figure:
    fig = go.Figure()
    xs = np.arange(max_state + 1, dtype=float)
    ys = np.zeros(max_state + 1)

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[str(i) for i in range(max_state + 1)],
        textposition="middle center",
        marker=dict(
            size=44,
            color=[COLORS[i % len(COLORS)] for i in range(max_state + 1)],
            line=dict(width=2, color="white"),
        ),
        hovertemplate="State %{text}<extra></extra>",
        showlegend=False,
    ))

    for n in range(max_state):
        lam_n = birth_rate(n, lam, finite, K)
        mu_next = total_service_rate(n + 1, thresholds, mu)

        if lam_n > 0:
            fig.add_annotation(
                x=n + 0.88, y=0.11, ax=n + 0.12, ay=0.11,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, text="",
            )
            fig.add_annotation(
                x=n + 0.5, y=0.23,
                text=f"λ = {lam_n:g}", showarrow=False, font=dict(size=12),
            )

        if mu_next > 0:
            fig.add_annotation(
                x=n + 0.12, y=-0.11, ax=n + 0.88, ay=-0.11,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, text="",
            )
            fig.add_annotation(
                x=n + 0.5, y=-0.23,
                text=f"μ{n+1} = {mu_next:g}", showarrow=False, font=dict(size=12),
            )

    if not finite:
        fig.add_annotation(
            x=max_state + 0.72, y=0, text="⋯", showarrow=False, font=dict(size=34),
        )
        fig.add_annotation(
            x=max_state + 0.95, y=0.14, ax=max_state + 0.25, ay=0.14,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            text=f"λ = {lam:g}", font=dict(size=11),
        )

    fig.update_layout(
        title="Birth-death transition-rate diagram",
        height=390,
        margin=dict(l=20, r=30, t=65, b=20),
        xaxis=dict(visible=False, range=[-0.6, max_state + (1.25 if not finite else 0.6)]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        showlegend=False,
    )
    return fig


def build_probability_figure(probabilities: np.ndarray, tail: float, finite: bool) -> go.Figure:
    states = [str(i) for i in range(len(probabilities))]
    values = list(probabilities)

    if not finite and tail > 1e-12:
        states.append(f"> {len(probabilities) - 1}")
        values.append(tail)

    fig = go.Figure(go.Bar(
        x=states,
        y=values,
        text=[f"{x:.5f}" for x in values],
        textposition="outside",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(values))],
    ))
    fig.update_layout(
        title="Steady-state probability distribution",
        xaxis_title="Number of customers in the system n",
        yaxis_title="Probability pₙ",
        yaxis=dict(range=[0, max(0.05, max(values) * 1.22)]),
        height=470,
        margin=dict(b=70),
    )
    return fig


def build_flow_balance_table(
    probabilities: np.ndarray,
    lam: float,
    mu: float,
    thresholds: list[int],
    finite: bool,
    K: int | None,
) -> pd.DataFrame:
    rows = []
    max_state = len(probabilities) - 1
    for n in range(max_state):
        lam_n = birth_rate(n, lam, finite, K)
        mu_next = total_service_rate(n + 1, thresholds, mu)
        left = probabilities[n] * lam_n
        right = probabilities[n + 1] * mu_next
        rows.append({
            "Transition": f"{n} ↔ {n+1}",
            "pₙ λₙ": left,
            "pₙ₊₁ μₙ₊₁": right,
            "Absolute difference": abs(left - right),
        })
    return pd.DataFrame(rows)


def fmt(value, decimals=6):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def metric_card(title: str, value: str, label: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Queue configuration")

model = st.sidebar.radio(
    "Queueing model",
    ["M/M/1", "M/M/s"],
    index=1,
)

capacity_type = st.sidebar.radio(
    "Queue capacity",
    ["Infinite", "Finite"],
    index=0,
)

lam = st.sidebar.number_input(
    "Arrival rate λ",
    min_value=0.000001,
    value=10.0,
    step=0.1,
    format="%.6f",
    help="Use a consistent time unit for both λ and μ.",
)

mu = st.sidebar.number_input(
    "Service rate μ per server",
    min_value=0.000001,
    value=5.0,
    step=0.1,
    format="%.6f",
    help="For example, a mean service time of 12 minutes corresponds to μ = 5 customers/hour.",
)

if model == "M/M/s":
    servers = int(st.sidebar.number_input(
        "Maximum number of servers s",
        min_value=2,
        max_value=50,
        value=3,
        step=1,
    ))

    activation_policy = st.sidebar.radio(
        "Server activation policy",
        ["Standard M/M/s", "State-dependent"],
        index=1,
        help=(
            "Standard M/M/s activates servers as customers arrive. State-dependent lets you define "
            "the population level at which each additional server is switched on."
        ),
    )

    if activation_policy == "Standard M/M/s":
        thresholds = list(range(1, servers + 1))
        st.sidebar.caption(
            "Standard policy: server j becomes useful from state n = j, so μₙ = min(n,s)μ."
        )
    else:
        st.sidebar.markdown("### Server activation thresholds")
        st.sidebar.caption(
            "Enter the first state n at which each server is switched on. "
            "Example: 1, 4, 7 gives 1 server for states 1–3, 2 for 4–6, and 3 from 7 onward."
        )
        thresholds = []
        for j in range(1, servers + 1):
            default = 1 if j == 1 else (4 if j == 2 and servers >= 2 else (7 if j == 3 and servers >= 3 else j))
            if j > 3:
                default = thresholds[-1] + 1
            value = int(st.sidebar.number_input(
                f"Activate server {j} from state n =",
                min_value=1,
                max_value=10000,
                value=default,
                step=1,
                key=f"activation_threshold_{j}_{servers}",
                disabled=(j == 1),
            ))
            thresholds.append(value)
else:
    servers = 1
    activation_policy = "Standard M/M/1"
    thresholds = [1]

finite = capacity_type == "Finite"

if finite:
    waiting_capacity = int(st.sidebar.number_input(
        "Finite waiting-room capacity",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
        help="Maximum number of customers that may wait, excluding the s service positions.",
    ))
    K = servers + waiting_capacity
    display_max_state = K
    st.sidebar.caption(f"Total system capacity: K = s + waiting places = {K}")
else:
    K = None
    try:
        R_preview = saturation_state(thresholds, servers)
    except Exception:
        R_preview = servers
    extra_states = int(st.sidebar.number_input(
        "States to display after service capacity is saturated",
        min_value=2,
        max_value=100,
        value=5,
        step=1,
        help=(
            "Only controls the displayed birth-death diagram and generator block. "
            "The stationary analysis remains an infinite-state calculation."
        ),
    ))
    display_max_state = R_preview + extra_states
    st.sidebar.caption(
        f"The current policy reaches constant total service capacity from state n = {R_preview}. "
        f"The display will show states 0 through {display_max_state}."
    )

st.sidebar.markdown("---")
solve = st.sidebar.button("Analyze queueing system", use_container_width=True, type="primary")


# ── Main header ──────────────────────────────────────────────────────────────
st.title("Analysis of Markovian Queueing Systems")
st.markdown(
    """
    <div class="info-box">
        This application models continuous-time Markovian queues as birth-death processes.
        In addition to standard M/M/1 and M/M/s systems, an M/M/s model may use a
        <b>state-dependent server activation policy</b>, allowing the number of operating
        servers to change according to the number of customers in the system.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Configuration signature ─────────────────────────────────────────────────
current_signature = (
    model,
    capacity_type,
    activation_policy,
    round(float(lam), 12),
    round(float(mu), 12),
    int(servers),
    tuple(int(x) for x in thresholds),
    int(K) if K is not None else None,
    int(display_max_state),
)


# ── Solve ────────────────────────────────────────────────────────────────────
if solve:
    try:
        if lam <= 0 or mu <= 0:
            raise ValueError("λ and μ must be positive.")

        validate_thresholds(thresholds, servers)
        R = saturation_state(thresholds, servers)

        if finite:
            if K < thresholds[-1]:
                st.warning(
                    f"The finite system ends at K = {K}, but the last server is scheduled to activate at "
                    f"n = {thresholds[-1]}. That server can therefore never be activated."
                )
            probabilities = finite_stationary_distribution(lam, mu, thresholds, K)
            tail = 0.0
            p0 = float(probabilities[0])
            metrics = finite_metrics(lam, mu, servers, thresholds, K, probabilities)
        else:
            if lam >= servers * mu:
                st.session_state.pop("queue_solution", None)
                st.error(
                    "The infinite-capacity system has no stationary distribution because, after all "
                    f"servers are active, λ = {lam:g} ≥ sμ = {servers * mu:g}."
                )
                st.stop()

            probabilities, tail, p0, R = infinite_stationary_distribution(
                lam, mu, servers, thresholds, display_max_state
            )
            metrics = infinite_metrics(lam, mu, servers, thresholds, p0)

        Q = build_generator_matrix(
            lam=lam,
            mu=mu,
            thresholds=thresholds,
            max_state=display_max_state,
            finite=finite,
            K=K,
        )

        rate_df = build_rate_table(
            lam=lam,
            mu=mu,
            thresholds=thresholds,
            max_state=display_max_state,
            finite=finite,
            K=K,
        )

        policy_df = build_policy_table(thresholds, finite, K)

        balance_df = build_flow_balance_table(
            probabilities=probabilities,
            lam=lam,
            mu=mu,
            thresholds=thresholds,
            finite=finite,
            K=K,
        )

        st.session_state["queue_solution"] = {
            "signature": current_signature,
            "probabilities": probabilities,
            "tail": tail,
            "metrics": metrics,
            "Q": Q,
            "rate_df": rate_df,
            "policy_df": policy_df,
            "balance_df": balance_df,
            "R": R,
        }
        st.success("Queueing system analyzed successfully.")

    except Exception as exc:
        st.session_state.pop("queue_solution", None)
        st.error(f"The queueing system could not be analyzed: {exc}")


# ── Recover solution ─────────────────────────────────────────────────────────
solution = st.session_state.get("queue_solution")
solution_is_valid = solution is not None and solution.get("signature") == current_signature

if solution_is_valid:
    probabilities = solution["probabilities"]
    tail = solution["tail"]
    metrics = solution["metrics"]
    Q = solution["Q"]
    rate_df = solution["rate_df"]
    policy_df = solution["policy_df"]
    balance_df = solution["balance_df"]
    R = solution["R"]
else:
    probabilities = tail = metrics = Q = rate_df = policy_df = balance_df = R = None


def require_solution_message():
    if solution is None:
        st.info("Configure the queue and click **Analyze queueing system** in the sidebar.")
    else:
        st.warning("The configuration changed. Click **Analyze queueing system** again.")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_bd, tab_q, tab_ss, tab_metrics, tab_probs, tab_balance = st.tabs([
    "Birth-death process",
    "Generator matrix Q",
    "Steady state",
    "Queue metrics",
    "State probabilities",
    "Flow balance",
])


# ── TAB 1: Birth-death process ───────────────────────────────────────────────
with tab_bd:
    st.markdown("## Birth-death structure")

    if model == "M/M/1":
        st.latex(r"\lambda_n=\lambda, \qquad \mu_n=\begin{cases}0,&n=0\\ \mu,&n\ge 1\end{cases}")
        st.info(
            "For M/M/1, the single server is active whenever at least one customer is present."
        )
    elif activation_policy == "Standard M/M/s":
        st.latex(r"\lambda_n=\lambda, \qquad \mu_n=\min(n,s)\mu")
        st.info(
            f"Standard M/M/s: service capacity rises with n until all {servers} servers are busy."
        )
    else:
        st.latex(r"\lambda_n=\lambda, \qquad \mu_n=b(n)\mu")
        st.info(
            "State-dependent policy: b(n) is the number of busy servers produced by the activation rule. "
            "The total service rate therefore changes at the activation thresholds rather than necessarily at n = 1,2,…,s."
        )

    if finite:
        st.latex(r"\lambda_K=0")
        st.caption(
            f"Finite-capacity system with K = {K}. State K is full, so new arrivals are blocked."
        )
    else:
        st.caption(
            "The state space is infinite. Only a finite leading portion is displayed; the stationary calculation uses the exact geometric tail after service capacity becomes constant."
        )

    if not solution_is_valid:
        require_solution_message()
    else:
        st.markdown("### Server activation policy")
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Total service capacity becomes constant from state n = {R}, where all {servers} servers are busy."
        )

        fig_bd = build_birth_death_figure(
            lam, mu, thresholds, display_max_state, finite, K
        )
        st.plotly_chart(fig_bd, use_container_width=True)

        st.markdown("### State-dependent transition rates")
        st.dataframe(rate_df, use_container_width=True, hide_index=True)


# ── TAB 2: Generator matrix Q ────────────────────────────────────────────────
with tab_q:
    st.markdown("## Infinitesimal generator matrix Q")
    st.latex(
        r"q_{n,n+1}=\lambda_n,\qquad q_{n,n-1}=\mu_n,\qquad "
        r"q_{n,n}=-(\lambda_n+\mu_n)"
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        state_labels = [str(i) for i in range(display_max_state + 1)]
        Q_df = pd.DataFrame(np.round(Q, 6), index=state_labels, columns=state_labels)
        st.dataframe(Q_df, use_container_width=True, height=520)

        if finite:
            row_sums = Q.sum(axis=1)
            max_error = float(np.max(np.abs(row_sums)))
            st.caption(
                f"Finite system: every row of Q sums to 0. Maximum numerical row-sum error = {max_error:.3e}."
            )
        else:
            st.warning(
                f"This is the leading block of the infinite generator, showing states 0,…,{display_max_state}. "
                f"The transition {display_max_state} → {display_max_state + 1} is outside the displayed block."
            )


# ── TAB 3: Steady state ─────────────────────────────────────────────────────
with tab_ss:
    st.markdown("## Steady-state analysis")
    st.latex(r"\boldsymbol{\pi}Q=0,\qquad \sum_n \pi_n=1")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")

    if not solution_is_valid:
        require_solution_message()
    else:
        if finite:
            st.success("The finite-state birth-death system has a stationary distribution.")
        else:
            st.success(
                f"Tail stability condition satisfied: λ/(sμ) = {metrics['rho']:.6f} < 1."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("P₀", fmt(metrics["p0"]), "Probability that the system is empty.")
        with c2:
            metric_card(
                "Tail traffic intensity",
                fmt(metrics["rho"]),
                "λ/(sμ) after all servers are active. For state-dependent systems this is a tail stability measure, not the utilization of each state.",
            )
        with c3:
            metric_card(
                "Mean busy servers",
                fmt(metrics["mean_busy_servers"]),
                "Expected number of servers simultaneously serving customers.",
            )

        st.markdown("### Stationary probabilities shown")
        p_df = pd.DataFrame({
            "State n": np.arange(len(probabilities)),
            "pₙ": probabilities,
        })
        st.dataframe(p_df, use_container_width=True, hide_index=True)

        if not finite:
            st.caption(
                f"Probability mass beyond state {display_max_state}: {tail:.8f}."
            )


# ── TAB 4: Queue metrics ─────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("## Queue performance metrics")

    if not solution_is_valid:
        require_solution_message()
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("L", fmt(metrics["L"]), "Expected number of customers in the system.")
        with c2:
            metric_card("Lq", fmt(metrics["Lq"]), "Expected number of customers waiting in queue.")
        with c3:
            metric_card("W", fmt(metrics["W"]), "Expected time a customer spends in the system.")
        with c4:
            metric_card("Wq", fmt(metrics["Wq"]), "Expected waiting time before service begins.")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            metric_card(
                "Effective arrival rate",
                fmt(metrics["lambda_effective"]),
                "Throughput λ_eff. For finite systems, blocked arrivals are excluded.",
            )
        with c6:
            metric_card(
                "Average server utilization",
                fmt(metrics["server_utilization"]),
                "Mean busy servers divided by the maximum number of servers s.",
            )
        with c7:
            metric_card(
                "Blocking probability",
                fmt(metrics["blocking_probability"]),
                "Probability that an arrival finds a finite-capacity system full.",
            )
        with c8:
            metric_card(
                "Probability of waiting",
                fmt(metrics["p_wait"]),
                "PASTA probability that an admitted arrival must wait after the server activation policy reacts.",
            )

        st.markdown("### Little's law checks")
        check_rows = [
            {
                "Relationship": "L = λ_eff W",
                "Left side": metrics["L"],
                "Right side": metrics["lambda_effective"] * metrics["W"],
            },
            {
                "Relationship": "Lq = λ_eff Wq",
                "Left side": metrics["Lq"],
                "Right side": metrics["lambda_effective"] * metrics["Wq"],
            },
        ]
        check_df = pd.DataFrame(check_rows)
        check_df["Absolute difference"] = np.abs(check_df["Left side"] - check_df["Right side"])
        st.dataframe(check_df, use_container_width=True, hide_index=True)


# ── TAB 5: State probabilities ───────────────────────────────────────────────
with tab_probs:
    st.markdown("## Probability distribution by system population")

    if not solution_is_valid:
        require_solution_message()
    else:
        fig_p = build_probability_figure(probabilities, tail, finite)
        st.plotly_chart(fig_p, use_container_width=True)

        prob_rows = []
        for n, pn in enumerate(probabilities):
            op = operating_servers(n, thresholds)
            busy = busy_servers(n, thresholds)
            prob_rows.append({
                "State n": n,
                "Servers in operation": op,
                "Busy servers": busy,
                "Customers waiting": max(n - busy, 0),
                "pₙ": pn,
            })
        prob_df = pd.DataFrame(prob_rows)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        if not finite and tail > 1e-12:
            st.info(
                f"The displayed bars omit individual states above {display_max_state}; their exact combined probability is {tail:.8f}."
            )


# ── TAB 6: Flow balance ──────────────────────────────────────────────────────
with tab_balance:
    st.markdown("## Birth-death flow-balance verification")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")
    st.info(
        "In steady state, the probability flow from n to n+1 equals the reverse flow from n+1 to n for every adjacent pair of states."
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
        max_balance_error = float(balance_df["Absolute difference"].max()) if not balance_df.empty else 0.0
        st.caption(f"Maximum local balance error in the displayed states: {max_balance_error:.3e}.")
