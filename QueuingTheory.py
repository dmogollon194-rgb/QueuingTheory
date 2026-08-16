import math
import hashlib
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(page_title="Markovian Queueing Systems", layout="wide")


# ── Constants ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"
TOL = 1e-8

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


# ── Helpers ──────────────────────────────────────────────────────────────────
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


def standard_generator_template(
    lam: float,
    mu: float,
    servers: int,
    max_state: int,
    finite: bool,
) -> np.ndarray:
    """
    Build an editable standard M/M/s generator template.

    For an infinite system, the last displayed row is a leading-block row of the
    infinite generator: the transition to state max_state + 1 is not displayed,
    but its rate is included in the diagonal.
    """
    n_states = max_state + 1
    Q = np.zeros((n_states, n_states), dtype=float)

    for n in range(n_states):
        birth = 0.0 if (finite and n == max_state) else lam
        death = min(n, servers) * mu

        if n + 1 < n_states and birth > 0:
            Q[n, n + 1] = birth
        if n > 0 and death > 0:
            Q[n, n - 1] = death

        Q[n, n] = -(birth + death)

    return Q


def parse_q_value(value) -> float:
    """Parse a Q-matrix entry written as a decimal or fraction."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Q contains empty cells.")

    text = str(value).strip()
    if text == "":
        raise ValueError("Q contains empty cells.")

    # Decimal comma is accepted when the file delimiter is not a comma
    # (for example, semicolon-separated CSV files).
    text = text.replace(",", ".")

    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except Exception as exc:
        raise ValueError(f"Invalid Q-matrix value: {value}") from exc


def coerce_editor_to_matrix(editor_df: pd.DataFrame) -> np.ndarray:
    Q = np.zeros(editor_df.shape, dtype=float)
    for i in range(editor_df.shape[0]):
        for j in range(editor_df.shape[1]):
            Q[i, j] = parse_q_value(editor_df.iat[i, j])

    if np.any(~np.isfinite(Q)):
        raise ValueError("Q contains empty or non-numeric cells.")
    return Q


def load_generator_matrix_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Read a headerless/indexless generator matrix from CSV or Excel.

    The file must contain only matrix values. Decimal values and fractions may
    be mixed. CSV delimiter detection is automatic.
    """
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    file_signature = hashlib.sha256(raw_bytes).hexdigest()[:16]

    try:
        if filename.endswith(".csv"):
            uploaded_file.seek(0)
            raw = pd.read_csv(
                uploaded_file,
                header=None,
                sep=None,
                engine="python",
                dtype=str,
            )
        elif filename.endswith(".xlsx"):
            uploaded_file.seek(0)
            raw = pd.read_excel(
                uploaded_file,
                header=None,
                dtype=object,
                engine="openpyxl",
            )
        elif filename.endswith(".xls"):
            uploaded_file.seek(0)
            raw = pd.read_excel(
                uploaded_file,
                header=None,
                dtype=object,
            )
        else:
            raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")
    except ImportError as exc:
        if filename.endswith(".xlsx"):
            raise ValueError(
                "Reading XLSX files requires openpyxl. Add 'openpyxl' to requirements.txt."
            ) from exc
        if filename.endswith(".xls"):
            raise ValueError(
                "Reading legacy XLS files may require xlrd. Add 'xlrd' to requirements.txt."
            ) from exc
        raise
    except Exception as exc:
        raise ValueError(f"The uploaded file could not be read: {exc}") from exc

    # Ignore completely empty exterior rows/columns, but not empty cells inside Q.
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if raw.empty:
        raise ValueError("The uploaded file is empty.")
    if raw.shape[0] != raw.shape[1]:
        raise ValueError(
            f"Q must be square. The uploaded matrix has shape {raw.shape[0]} × {raw.shape[1]}."
        )
    if raw.shape[0] < 2:
        raise ValueError("Q must contain at least states 0 and 1.")
    if raw.shape[0] > 101:
        raise ValueError("The uploaded matrix may contain at most 101 states (0 through 100).")
    if raw.isna().any().any():
        raise ValueError(
            "The uploaded matrix contains empty cells. The file must contain only the complete Q matrix."
        )

    Q = np.zeros(raw.shape, dtype=float)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            Q[i, j] = parse_q_value(raw.iat[i, j])

    labels = [str(i) for i in range(Q.shape[0])]
    return pd.DataFrame(Q, index=labels, columns=labels), file_signature


def validate_generator(Q: np.ndarray, finite: bool):
    """
    Validate a birth-death infinitesimal generator.

    Finite system:
        every row must sum to zero.

    Infinite system:
        rows 0,...,N-1 must sum to zero.
        row N is the final visible row of an infinite leading block. Its missing
        transition N -> N+1 is recovered from the diagonal:
            lambda_tail = -q_NN - q_N,N-1.
        That last pair of birth/death rates is assumed to repeat for n >= N.
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    m = Q.shape[0]
    if m < 2:
        raise ValueError("Q must contain at least states 0 and 1.")

    # Off-diagonal generator entries must be nonnegative.
    for i in range(m):
        for j in range(m):
            value = Q[i, j]
            if i != j and value < -TOL:
                raise ValueError(
                    f"Off-diagonal generator entries must be nonnegative. "
                    f"Invalid value q[{i},{j}] = {value:g}."
                )
            if abs(i - j) > 1 and abs(value) > TOL:
                raise ValueError(
                    "This application analyzes birth-death queues, so only "
                    "transitions between adjacent states are allowed. "
                    f"q[{i},{j}] must be 0."
                )

        if Q[i, i] > TOL:
            raise ValueError(f"Diagonal entry q[{i},{i}] must be nonpositive.")

    row_sums = Q.sum(axis=1)

    if finite:
        if not np.allclose(row_sums, 0.0, atol=1e-7):
            bad = np.where(~np.isclose(row_sums, 0.0, atol=1e-7))[0].tolist()
            raise ValueError(
                f"For a finite generator, every row must sum to 0. Check rows {bad}."
            )
        lambda_tail = None
        mu_tail = None
    else:
        if not np.allclose(row_sums[:-1], 0.0, atol=1e-7):
            bad = np.where(~np.isclose(row_sums[:-1], 0.0, atol=1e-7))[0].tolist()
            raise ValueError(
                f"Rows 0 through N-1 of the displayed infinite-generator block must sum to 0. "
                f"Check rows {bad}."
            )

        N = m - 1
        mu_tail = float(Q[N, N - 1])
        lambda_tail = float(-Q[N, N] - mu_tail)

        if mu_tail <= TOL:
            raise ValueError(
                "The final visible state must have a positive death/service rate."
            )
        if lambda_tail <= TOL:
            raise ValueError(
                "For an infinite queue, the final row must include the unshown transition "
                "to the next state through its diagonal. The implied tail birth rate must be positive."
            )

        expected_last_sum = -lambda_tail
        if not np.isclose(row_sums[-1], expected_last_sum, atol=1e-7):
            raise ValueError("The final row of the infinite generator block is inconsistent.")

    # Positive rates are required along the birth-death path.
    last = m - 1
    max_birth_row = last - 1 if finite else last - 1
    for n in range(max_birth_row + 1):
        if Q[n, n + 1] <= TOL:
            raise ValueError(f"Birth rate q[{n},{n+1}] must be positive.")

    for n in range(1, m):
        if Q[n, n - 1] <= TOL:
            raise ValueError(f"Death/service rate q[{n},{n-1}] must be positive.")

    return lambda_tail, mu_tail, row_sums


def extract_rates(Q: np.ndarray, finite: bool):
    m = Q.shape[0]
    N = m - 1

    births = np.zeros(m, dtype=float)
    deaths = np.zeros(m, dtype=float)

    for n in range(m):
        if n < N:
            births[n] = max(float(Q[n, n + 1]), 0.0)
        if n > 0:
            deaths[n] = max(float(Q[n, n - 1]), 0.0)

    if finite:
        births[N] = 0.0
        lambda_tail = None
        mu_tail = None
    else:
        mu_tail = deaths[N]
        lambda_tail = float(-Q[N, N] - mu_tail)
        births[N] = lambda_tail

    return births, deaths, lambda_tail, mu_tail


def infer_busy_servers(deaths: np.ndarray, mu_per_server: float, servers: int) -> np.ndarray:
    """Infer the number of busy identical exponential servers from mu_n / mu."""
    if mu_per_server <= 0:
        raise ValueError("Service rate μ per server must be positive.")

    raw = deaths / mu_per_server
    rounded = np.rint(raw)

    for n, (x, r) in enumerate(zip(raw, rounded)):
        if n == 0:
            if abs(x) > 1e-7:
                raise ValueError("State 0 cannot have a positive service/death rate.")
            continue
        if not np.isclose(x, r, atol=1e-6):
            raise ValueError(
                f"The death rate in state {n} is μ_{n} = {deaths[n]:g}, which is not an integer "
                f"multiple of the per-server rate μ = {mu_per_server:g}."
            )
        if r < 1 or r > servers:
            raise ValueError(
                f"State {n} implies {int(r)} busy servers, outside the allowed range 1,...,{servers}."
            )
        if r > n:
            raise ValueError(
                f"State {n} cannot have {int(r)} busy servers because only {n} customers are present."
            )

    return rounded.astype(int)


def validate_mm_arrivals(
    births: np.ndarray,
    lam_reference: float,
    finite: bool,
    lambda_tail: float | None,
):
    """
    M/M/s assumes a state-independent Poisson arrival rate while the system is
    able to accept customers. Q is still the calculation source; lambda is used
    here only as a consistency check against the problem data.
    """
    if lam_reference <= 0:
        raise ValueError("Arrival rate λ must be positive.")

    if finite:
        check = births[:-1]
    else:
        check = births[:-1]

    if not np.allclose(check, lam_reference, atol=1e-6, rtol=1e-6):
        bad = np.where(~np.isclose(check, lam_reference, atol=1e-6, rtol=1e-6))[0].tolist()
        raise ValueError(
            f"For an M/M/s queue, accepted arrivals must have the same Poisson rate λ = {lam_reference:g}. "
            f"Check Q birth rates in rows {bad}."
        )

    if not finite and lambda_tail is not None and not np.isclose(
        lambda_tail, lam_reference, atol=1e-6, rtol=1e-6
    ):
        raise ValueError(
            f"The final row implies a tail arrival rate {lambda_tail:g}, but λ = {lam_reference:g}."
        )


def finite_stationary_from_q(births: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    K = len(births) - 1
    weights = np.ones(K + 1, dtype=float)

    for n in range(1, K + 1):
        if deaths[n] <= TOL:
            raise ValueError(f"Death rate μ_{n} must be positive.")
        weights[n] = weights[n - 1] * births[n - 1] / deaths[n]

    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("The stationary distribution could not be normalized.")

    return weights / total


def infinite_stationary_from_q(
    births: np.ndarray,
    deaths: np.ndarray,
    lambda_tail: float,
    mu_tail: float,
):
    """
    Exact infinite stationary distribution when the final visible row defines
    the repeating birth/death rates for the geometric tail.
    """
    N = len(births) - 1
    r = lambda_tail / mu_tail

    if r >= 1.0 - 1e-12:
        raise ValueError(
            f"The infinite queue has no stationary distribution because the repeating tail has "
            f"λ_tail / μ_tail = {r:.6f} ≥ 1."
        )

    weights = np.ones(N + 1, dtype=float)
    for n in range(1, N + 1):
        if deaths[n] <= TOL:
            raise ValueError(f"Death rate μ_{n} must be positive.")
        weights[n] = weights[n - 1] * births[n - 1] / deaths[n]

    # States 0,...,N-1 explicitly; states N,N+1,... form the repeating tail.
    normalizer = float(weights[:N].sum() + weights[N] / (1.0 - r))
    p0 = 1.0 / normalizer
    probs = p0 * weights
    tail_beyond = float(probs[N] * r / (1.0 - r))

    return probs, tail_beyond, r


def finite_metrics_from_q(
    births: np.ndarray,
    deaths: np.ndarray,
    p: np.ndarray,
    busy: np.ndarray,
    servers: int,
):
    states = np.arange(len(p), dtype=float)
    waiting = np.maximum(states - busy, 0.0)

    L = float(np.sum(states * p))
    Lq = float(np.sum(waiting * p))
    mean_busy = float(np.sum(busy * p))

    lambda_eff = float(np.sum(p * births))
    W = L / lambda_eff if lambda_eff > TOL else np.nan
    Wq = Lq / lambda_eff if lambda_eff > TOL else np.nan

    # Arrival-rate-weighted probability that an admitted arrival must wait.
    wait_flow = 0.0
    for n in range(len(p) - 1):
        busy_after_arrival = busy[n + 1]
        if n + 1 > busy_after_arrival:
            wait_flow += p[n] * births[n]
    p_wait = wait_flow / lambda_eff if lambda_eff > TOL else np.nan

    return {
        "rho_tail": None,
        "p0": float(p[0]),
        "p_wait": float(p_wait),
        "blocking_probability": float(p[-1]),
        "lambda_effective": lambda_eff,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": mean_busy,
        "server_utilization": mean_busy / servers,
    }


def infinite_metrics_from_q(
    births: np.ndarray,
    deaths: np.ndarray,
    p: np.ndarray,
    busy: np.ndarray,
    servers: int,
    lambda_tail: float,
    mu_tail: float,
):
    N = len(p) - 1
    r = lambda_tail / mu_tail
    pN = float(p[N])

    busy_tail = int(busy[N])

    # Exact infinite sums. Tail starts at state N.
    L_prefix = float(sum(n * p[n] for n in range(N)))
    L_tail = pN * (N / (1.0 - r) + r / ((1.0 - r) ** 2))
    L = float(L_prefix + L_tail)

    Lq_prefix = float(sum(max(n - busy[n], 0) * p[n] for n in range(N)))
    Lq_tail = pN * (
        max(N - busy_tail, 0) / (1.0 - r)
        + r / ((1.0 - r) ** 2)
    )
    Lq = float(Lq_prefix + Lq_tail)

    busy_prefix = float(sum(busy[n] * p[n] for n in range(N)))
    mean_busy = float(busy_prefix + busy_tail * pN / (1.0 - r))

    lambda_prefix = float(sum(p[n] * births[n] for n in range(N)))
    lambda_tail_flow = lambda_tail * pN / (1.0 - r)
    lambda_eff = float(lambda_prefix + lambda_tail_flow)

    W = L / lambda_eff if lambda_eff > TOL else np.nan
    Wq = Lq / lambda_eff if lambda_eff > TOL else np.nan

    wait_flow_prefix = 0.0
    for n in range(N):
        if n + 1 > busy[n + 1]:
            wait_flow_prefix += p[n] * births[n]

    tail_waits = (N + 1) > busy_tail
    wait_flow_tail = lambda_tail_flow if tail_waits else 0.0
    p_wait = (wait_flow_prefix + wait_flow_tail) / lambda_eff if lambda_eff > TOL else np.nan

    return {
        "rho_tail": r,
        "p0": float(p[0]),
        "p_wait": float(p_wait),
        "blocking_probability": 0.0,
        "lambda_effective": lambda_eff,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": mean_busy,
        "server_utilization": mean_busy / servers,
    }


def build_rate_table(
    births: np.ndarray,
    deaths: np.ndarray,
    busy: np.ndarray,
    finite: bool,
) -> pd.DataFrame:
    rows = []
    N = len(births) - 1

    for n in range(len(births)):
        rows.append({
            "State n": n,
            "Busy servers inferred from Q": int(busy[n]),
            "Customers waiting": max(int(n - busy[n]), 0),
            "Birth rate λₙ": float(births[n]),
            "Death rate μₙ": float(deaths[n]),
            "Note": (
                "Full state: arrivals blocked" if finite and n == N
                else "Repeating tail starts here" if (not finite and n == N)
                else ""
            ),
        })

    return pd.DataFrame(rows)


def build_birth_death_figure(
    births: np.ndarray,
    deaths: np.ndarray,
    finite: bool,
) -> go.Figure:
    N = len(births) - 1
    fig = go.Figure()
    xs = np.arange(N + 1, dtype=float)
    ys = np.zeros(N + 1)

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[str(i) for i in range(N + 1)],
        textposition="middle center",
        marker=dict(
            size=44,
            color=[COLORS[i % len(COLORS)] for i in range(N + 1)],
            line=dict(width=2, color="white"),
        ),
        hovertemplate="State %{text}<extra></extra>",
        showlegend=False,
    ))

    for n in range(N):
        lam_n = births[n]
        mu_next = deaths[n + 1]

        fig.add_annotation(
            x=n + 0.88, y=0.11, ax=n + 0.12, ay=0.11,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, text="",
        )
        fig.add_annotation(
            x=n + 0.5, y=0.23,
            text=f"λ{n} = {lam_n:g}", showarrow=False, font=dict(size=12),
        )

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
        lambda_tail = births[N]
        mu_tail = deaths[N]

        fig.add_annotation(
            x=N + 0.72, y=0, text="⋯", showarrow=False, font=dict(size=34),
        )
        fig.add_annotation(
            x=N + 0.95, y=0.14, ax=N + 0.25, ay=0.14,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            text=f"λ = {lambda_tail:g}", font=dict(size=11),
        )
        fig.add_annotation(
            x=N + 0.25, y=-0.14, ax=N + 0.95, ay=-0.14,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
            text=f"μ = {mu_tail:g}", font=dict(size=11),
        )

    fig.update_layout(
        title="Birth-death transition-rate diagram derived from Q",
        height=390,
        margin=dict(l=20, r=30, t=65, b=20),
        xaxis=dict(visible=False, range=[-0.6, N + (1.35 if not finite else 0.6)]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        showlegend=False,
    )
    return fig


def build_probability_figure(probabilities: np.ndarray, tail_beyond: float, finite: bool):
    states = [str(i) for i in range(len(probabilities))]
    values = list(probabilities)

    if not finite and tail_beyond > 1e-12:
        states.append(f"> {len(probabilities) - 1}")
        values.append(tail_beyond)

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
    births: np.ndarray,
    deaths: np.ndarray,
) -> pd.DataFrame:
    rows = []
    N = len(probabilities) - 1

    for n in range(N):
        left = probabilities[n] * births[n]
        right = probabilities[n + 1] * deaths[n + 1]
        rows.append({
            "Transition": f"{n} ↔ {n+1}",
            "pₙ λₙ": left,
            "pₙ₊₁ μₙ₊₁": right,
            "Absolute difference": abs(left - right),
        })

    return pd.DataFrame(rows)


# ── Sidebar configuration ────────────────────────────────────────────────────
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
    help="Problem-data arrival rate. It is used to initialize and validate Q.",
)

mu = st.sidebar.number_input(
    "Service rate μ per server",
    min_value=0.000001,
    value=5.0,
    step=0.1,
    format="%.6f",
    help="Per-server service rate. For example, a mean service time of 12 minutes gives μ = 5 customers/hour.",
)

if model == "M/M/s":
    servers = int(st.sidebar.number_input(
        "Number of servers s",
        min_value=2,
        max_value=50,
        value=3,
        step=1,
    ))
else:
    servers = 1

finite = capacity_type == "Finite"

st.sidebar.markdown("---")
st.sidebar.markdown("### Generator matrix Q input")
q_input_mode = st.sidebar.radio(
    "Matrix source",
    ["Build / edit in the app", "Load CSV / Excel"],
    index=0,
    help=(
        "The uploaded file must contain only the Q values, with no row names, "
        "column names, or headers."
    ),
)

uploaded_q_file = None
uploaded_q_df = None
upload_signature = None
upload_error = None

if q_input_mode == "Load CSV / Excel":
    uploaded_q_file = st.sidebar.file_uploader(
        "Upload generator matrix Q",
        type=["csv", "xlsx", "xls"],
        help=(
            "Use a square matrix containing only values. CSV and Excel files are read "
            "without headers or index columns."
        ),
    )

    if uploaded_q_file is not None:
        try:
            uploaded_q_df, upload_signature = load_generator_matrix_file(uploaded_q_file)
        except Exception as exc:
            upload_error = str(exc)

if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
    max_state = uploaded_q_df.shape[0] - 1
    K = max_state if finite else None

    if finite:
        inferred_waiting_capacity = max(max_state - servers, 0)
        st.sidebar.caption(
            f"Uploaded Q defines states 0,...,{max_state}; therefore total system capacity is K = {max_state}. "
            f"With s = {servers}, the implied waiting-room capacity is {inferred_waiting_capacity}."
        )
    else:
        st.sidebar.caption(
            f"Uploaded Q defines the visible states 0,...,{max_state}. The final row is treated as "
            "the first state of the repeating infinite tail."
        )
else:
    if finite:
        waiting_capacity = int(st.sidebar.number_input(
            "Finite waiting-room capacity",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            help="Maximum number of waiting customers, excluding service positions.",
        ))
        K = servers + waiting_capacity
        max_state = K
        st.sidebar.caption(f"Total system capacity: K = {K}. Q has {K + 1} states (0,...,{K}).")
    else:
        K = None
        max_state = int(st.sidebar.number_input(
            "Last state represented explicitly in Q",
            min_value=max(2, servers),
            max_value=100,
            value=max(7, servers + 3),
            step=1,
            help=(
                "For an infinite queue, Q shows a finite leading block. The final row is treated as the "
                "first state of the repeating tail, so choose it at a state from which the birth and death "
                "rates remain unchanged."
            ),
        ))
        st.sidebar.caption(
            f"Q will show states 0,...,{max_state}. The last row also encodes the unshown transition "
            f"{max_state} → {max_state + 1} through its diagonal."
        )

if upload_error is not None:
    st.sidebar.error(upload_error)

st.sidebar.markdown("---")
solve_clicked = st.sidebar.button(
    "Analyze queueing system",
    use_container_width=True,
    type="primary",
    disabled=(q_input_mode == "Load CSV / Excel" and uploaded_q_df is None),
)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("Analysis of Markovian Queueing Systems")
st.markdown(
    """
    <div class="info-box">
        The queue is modeled as a continuous-time birth-death process. The editable
        <b>generator matrix Q is the calculation input</b>. Q may be built directly in the app or
        loaded from a CSV/Excel file containing only matrix values. The queue parameters define the
        problem context; all transition rates used in the analysis are read directly from Q.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_bd, tab_q, tab_ss, tab_metrics, tab_probs, tab_balance = st.tabs([
    "Birth-death process",
    "Generator matrix Q",
    "Steady state",
    "Queue metrics",
    "State probabilities",
    "Flow balance",
])


# ── Q editor / file-loaded Q: rendered before solving ────────────────────────
if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
    source_token = f"upload_{upload_signature}"
else:
    source_token = "manual"

base_key = f"q_base_{source_token}_{capacity_type}_{model}_{servers}_{max_state}"
reset_counter_key = f"q_reset_counter_{source_token}_{capacity_type}_{model}_{servers}_{max_state}"

if base_key not in st.session_state:
    labels = [str(i) for i in range(max_state + 1)]
    if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
        st.session_state[base_key] = uploaded_q_df.copy()
    else:
        template = standard_generator_template(lam, mu, servers, max_state, finite)
        st.session_state[base_key] = pd.DataFrame(template, index=labels, columns=labels)

if reset_counter_key not in st.session_state:
    st.session_state[reset_counter_key] = 0

with tab_q:
    st.markdown("## Infinitesimal generator matrix Q")
    st.latex(
        r"q_{n,n+1}=\lambda_n,\qquad q_{n,n-1}=\mu_n,\qquad "
        r"q_{n,n}=-\sum_{j\ne n}q_{n,j}"
    )

    if q_input_mode == "Load CSV / Excel":
        st.info(
            "Q is loaded from the selected CSV/Excel file. The file must contain only the square matrix "
            "of values: no title row, no state-name column, and no headers. After loading, the matrix is "
            "shown in the editor and may still be adjusted before analysis."
        )
    else:
        st.info(
            "Edit Q directly. For a birth-death queue, only the main diagonal and the two adjacent "
            "diagonals may contain nonzero values. All off-diagonal transition rates must be nonnegative."
        )

    if finite:
        st.caption(
            "Finite system: every row of Q must sum to 0. The final state is full, so its birth rate is 0."
        )
    else:
        st.caption(
            f"Infinite system: rows 0,...,{max_state - 1} must sum to 0. In the final visible row "
            f"n = {max_state}, the transition to state {max_state + 1} is outside the displayed matrix; "
            "its birth rate is recovered from the diagonal. The birth and death rates of this final row "
            "are then repeated for the infinite tail."
        )

    c_reset, c_note = st.columns([1, 3])
    with c_reset:
        if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
            reset_q = st.button("Reload Q from file", use_container_width=True)
        else:
            reset_q = st.button("Reset Q to standard M/M/s", use_container_width=True)
    with c_note:
        if q_input_mode == "Load CSV / Excel":
            st.caption(
                "Reload discards edits made after the file was loaded and restores the uploaded matrix."
            )
        else:
            st.caption(
                "Reset uses the current λ, μ and s. Manual edits are preserved on normal Streamlit reruns."
            )

    if reset_q:
        labels = [str(i) for i in range(max_state + 1)]
        if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
            st.session_state[base_key] = uploaded_q_df.copy()
        else:
            template = standard_generator_template(lam, mu, servers, max_state, finite)
            st.session_state[base_key] = pd.DataFrame(template, index=labels, columns=labels)
        st.session_state[reset_counter_key] += 1

    editor_key = (
        f"q_editor_{source_token}_{capacity_type}_{model}_{servers}_{max_state}_"
        f"{st.session_state[reset_counter_key]}"
    )

    edited_q_df = st.data_editor(
        st.session_state[base_key],
        use_container_width=True,
        height=min(620, 80 + 36 * (max_state + 1)),
        num_rows="fixed",
        key=editor_key,
    )

    if q_input_mode == "Load CSV / Excel" and uploaded_q_df is not None:
        st.success(
            f"Loaded {uploaded_q_df.shape[0]} × {uploaded_q_df.shape[1]} Q matrix from "
            f"{uploaded_q_file.name}. The number of represented states was detected automatically."
        )

    st.caption(
        "All calculations below derive λₙ and μₙ from the Q currently displayed in this editor."
    )


# ── Read the current Q and create a signature ────────────────────────────────
Q_input = None
q_input_error = None
try:
    Q_input = coerce_editor_to_matrix(edited_q_df)
except Exception as exc:
    q_input_error = str(exc)

if Q_input is not None:
    q_signature = tuple(np.round(Q_input.flatten(), 10).tolist())
else:
    q_signature = ("invalid",)

current_signature = (
    model,
    capacity_type,
    q_input_mode,
    upload_signature if q_input_mode == "Load CSV / Excel" else None,
    round(float(lam), 10),
    round(float(mu), 10),
    int(servers),
    int(max_state),
    q_signature,
)


# ── Solve using Q as the source of transition rates ─────────────────────────
if solve_clicked:
    try:
        if q_input_error is not None:
            raise ValueError(q_input_error)

        Q = Q_input.copy()
        lambda_tail, mu_tail, row_sums = validate_generator(Q, finite)
        births, deaths, lambda_tail, mu_tail = extract_rates(Q, finite)

        # M/M/s consistency checks. They do not generate the rates; Q already did.
        validate_mm_arrivals(births, lam, finite, lambda_tail)
        busy = infer_busy_servers(deaths, mu, servers)

        if finite:
            probabilities = finite_stationary_from_q(births, deaths)
            tail_beyond = 0.0
            metrics = finite_metrics_from_q(
                births=births,
                deaths=deaths,
                p=probabilities,
                busy=busy,
                servers=servers,
            )
            tail_ratio = None
        else:
            probabilities, tail_beyond, tail_ratio = infinite_stationary_from_q(
                births=births,
                deaths=deaths,
                lambda_tail=lambda_tail,
                mu_tail=mu_tail,
            )
            metrics = infinite_metrics_from_q(
                births=births,
                deaths=deaths,
                p=probabilities,
                busy=busy,
                servers=servers,
                lambda_tail=lambda_tail,
                mu_tail=mu_tail,
            )

        rate_df = build_rate_table(births, deaths, busy, finite)
        balance_df = build_flow_balance_table(probabilities, births, deaths)

        st.session_state["queue_solution_q"] = {
            "signature": current_signature,
            "Q": Q,
            "births": births,
            "deaths": deaths,
            "busy": busy,
            "probabilities": probabilities,
            "tail_beyond": tail_beyond,
            "metrics": metrics,
            "rate_df": rate_df,
            "balance_df": balance_df,
            "row_sums": row_sums,
            "lambda_tail": lambda_tail,
            "mu_tail": mu_tail,
            "tail_ratio": tail_ratio,
        }
        st.success("Queueing system analyzed successfully from the generator matrix Q currently displayed in the editor.")

    except Exception as exc:
        st.session_state.pop("queue_solution_q", None)
        st.error(f"The queueing system could not be analyzed: {exc}")


# ── Recover solution ─────────────────────────────────────────────────────────
solution = st.session_state.get("queue_solution_q")
solution_is_valid = solution is not None and solution.get("signature") == current_signature

if solution_is_valid:
    Q = solution["Q"]
    births = solution["births"]
    deaths = solution["deaths"]
    busy = solution["busy"]
    probabilities = solution["probabilities"]
    tail_beyond = solution["tail_beyond"]
    metrics = solution["metrics"]
    rate_df = solution["rate_df"]
    balance_df = solution["balance_df"]
    row_sums = solution["row_sums"]
    lambda_tail = solution["lambda_tail"]
    mu_tail = solution["mu_tail"]
    tail_ratio = solution["tail_ratio"]
else:
    Q = births = deaths = busy = probabilities = None
    tail_beyond = metrics = rate_df = balance_df = row_sums = None
    lambda_tail = mu_tail = tail_ratio = None


def require_solution_message():
    if solution is None:
        st.info("Edit Q and click **Analyze queueing system** in the sidebar.")
    else:
        st.warning("Q or the queue configuration changed. Click **Analyze queueing system** again.")


# ── Finish TAB Q with validation information ─────────────────────────────────
with tab_q:
    st.markdown("---")
    st.markdown("### Matrix used by the current analysis")

    if not solution_is_valid:
        require_solution_message()
    else:
        labels = [str(i) for i in range(max_state + 1)]
        Q_df = pd.DataFrame(np.round(Q, 6), index=labels, columns=labels)
        st.dataframe(Q_df, use_container_width=True, height=520)

        if finite:
            st.success(
                f"Valid finite generator. Maximum absolute row-sum error: "
                f"{np.max(np.abs(row_sums)):.3e}."
            )
        else:
            st.success(
                f"Valid leading block of an infinite generator. The final row implies "
                f"λ_tail = {lambda_tail:g}, μ_tail = {mu_tail:g}, and "
                f"λ_tail/μ_tail = {tail_ratio:.6f}."
            )
            st.caption(
                f"The final visible row sums to {-lambda_tail:g} because the transition "
                f"{max_state} → {max_state + 1} lies outside the displayed block."
            )


# ── TAB 1: Birth-death process ───────────────────────────────────────────────
with tab_bd:
    st.markdown("## Birth-death structure derived from Q")
    st.latex(r"\lambda_n=q_{n,n+1},\qquad \mu_n=q_{n,n-1}")
    st.info(
        "The diagram and the state-dependent rates are extracted from the edited generator matrix. "
        "No separate server-activation policy is entered."
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        fig_bd = build_birth_death_figure(births, deaths, finite)
        st.plotly_chart(fig_bd, use_container_width=True)

        st.markdown("### Transition rates implied by Q")
        st.dataframe(rate_df, use_container_width=True, hide_index=True)


# ── TAB 3: Steady state ─────────────────────────────────────────────────────
with tab_ss:
    st.markdown("## Steady-state analysis")
    st.latex(r"\boldsymbol{\pi}Q=0,\qquad \sum_n \pi_n=1")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")

    if not solution_is_valid:
        require_solution_message()
    else:
        if finite:
            st.success("The finite birth-death generator has been normalized to a stationary distribution.")
        else:
            st.success(
                f"The repeating tail is stable because λ_tail/μ_tail = {tail_ratio:.6f} < 1."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("P₀", fmt(metrics["p0"]), "Probability that the system is empty.")
        with c2:
            metric_card(
                "Tail ratio" if not finite else "States",
                fmt(tail_ratio) if not finite else str(len(probabilities)),
                "λ_tail/μ_tail for the repeating infinite tail."
                if not finite else "Number of states in the finite CTMC.",
            )
        with c3:
            metric_card(
                "Mean busy servers",
                fmt(metrics["mean_busy_servers"]),
                "Expected number of simultaneously busy servers inferred from Q.",
            )

        p_df = pd.DataFrame({
            "State n": np.arange(len(probabilities)),
            "pₙ": probabilities,
        })
        st.dataframe(p_df, use_container_width=True, hide_index=True)

        if not finite:
            st.caption(
                f"Combined probability beyond state {max_state}: {tail_beyond:.8f}."
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
                "Throughput computed from the stationary probabilities and Q birth rates.",
            )
        with c6:
            metric_card(
                "Average server utilization",
                fmt(metrics["server_utilization"]),
                "Mean busy servers divided by s.",
            )
        with c7:
            metric_card(
                "Blocking probability",
                fmt(metrics["blocking_probability"]),
                "Probability an arrival finds a finite-capacity system full.",
            )
        with c8:
            metric_card(
                "Probability of waiting",
                fmt(metrics["p_wait"]),
                "Arrival-rate-weighted probability that an admitted customer must wait.",
            )

        st.markdown("### Little's law checks")
        check_df = pd.DataFrame([
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
        ])
        check_df["Absolute difference"] = np.abs(check_df["Left side"] - check_df["Right side"])
        st.dataframe(check_df, use_container_width=True, hide_index=True)


# ── TAB 5: State probabilities ───────────────────────────────────────────────
with tab_probs:
    st.markdown("## Probability distribution by system population")

    if not solution_is_valid:
        require_solution_message()
    else:
        fig_p = build_probability_figure(probabilities, tail_beyond, finite)
        st.plotly_chart(fig_p, use_container_width=True)

        prob_rows = []
        for n, pn in enumerate(probabilities):
            prob_rows.append({
                "State n": n,
                "Busy servers inferred from Q": int(busy[n]),
                "Customers waiting": max(int(n - busy[n]), 0),
                "pₙ": pn,
            })
        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

        if not finite and tail_beyond > 1e-12:
            st.info(
                f"States above {max_state} are not shown individually. Their exact combined "
                f"stationary probability is {tail_beyond:.8f}."
            )


# ── TAB 6: Flow balance ──────────────────────────────────────────────────────
with tab_balance:
    st.markdown("## Birth-death flow-balance verification")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")
    st.info(
        "The local balance equations are checked using the transition rates read directly from Q."
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
        max_balance_error = (
            float(balance_df["Absolute difference"].max()) if not balance_df.empty else 0.0
        )
        st.caption(f"Maximum local balance error in the displayed states: {max_balance_error:.3e}.")
