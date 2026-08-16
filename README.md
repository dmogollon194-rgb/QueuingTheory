# Markovian Queueing Systems

Interactive application developed with **Python** and **Streamlit** for the analysis of Markovian queueing systems using continuous-time birth-death processes.

The application automatically constructs the transition-rate structure and the infinitesimal generator matrix \(Q\) from the parameters of the queueing system.

## Supported Models

The application currently supports:

- M/M/1 with infinite queue
- M/M/1 with finite queue
- M/M/s with infinite queue
- M/M/s with finite queue

For M/M/s systems, the user can specify the number of servers \(s\).

## Main Features

The application allows the user to:

- Define the arrival rate \(\lambda\)
- Define the service rate \(\mu\)
- Specify the number of servers \(s\)
- Define the waiting-room capacity for finite queueing systems
- Visualize the birth-death process
- Automatically construct the infinitesimal generator matrix \(Q\)
- Compute steady-state probabilities
- Analyze system utilization
- Compute queueing performance measures
- Analyze flow balance between states
- Visualize state probabilities

## Birth-Death Process

For a Markovian queueing system, transitions occur between neighboring states.

The arrival rate is:

\[
\lambda_n = \lambda
\]

for states where new arrivals are accepted.

For an M/M/s system, the total service rate is:

\[
\mu_n = \min(n,s)\mu
\]

Therefore:

- If \(n < s\), only \(n\) servers are busy.
- If \(n \geq s\), all \(s\) servers are busy.
- The maximum total service rate is \(s\mu\).

For example, with three servers:

\[
\mu_1 = \mu
\]

\[
\mu_2 = 2\mu
\]

\[
\mu_3 = 3\mu
\]

and

\[
\mu_n = 3\mu, \qquad n \geq 3
\]

## Stability Condition

For infinite-capacity M/M/s systems, a stationary distribution exists when:

\[
\rho = \frac{\lambda}{s\mu} < 1
\]

or equivalently:

\[
\lambda < s\mu
\]

This condition means that the long-run arrival rate must be smaller than the maximum service capacity of the system.

## Generator Matrix

The application automatically constructs the infinitesimal generator matrix \(Q\).

For a birth-death process:

\[
q_{n,n+1} = \lambda_n
\]

\[
q_{n,n-1} = \mu_n
\]

and the diagonal elements are:

\[
q_{n,n}=-(\lambda_n+\mu_n)
\]

so that every complete row of the generator matrix satisfies:

\[
\sum_j q_{ij}=0
\]

For infinite queueing systems, only a finite block of the infinite generator matrix is displayed for visualization purposes.

## Queueing Performance Measures

The application computes several standard performance measures.

### Average number of customers in the system

\[
L
\]

### Average number of customers waiting in queue

\[
L_q
\]

### Average time spent in the system

\[
W
\]

### Average waiting time in queue

\[
W_q
\]

Little's Law is used when applicable:

\[
L = \lambda_{\text{eff}} W
\]

and

\[
L_q = \lambda_{\text{eff}} W_q
\]

where \(\lambda_{\text{eff}}\) represents the effective arrival rate.

For finite-capacity systems:

\[
\lambda_{\text{eff}} = \lambda(1-p_K)
\]

where \(p_K\) is the probability that the system is full.

## Finite-Capacity Systems

For finite systems, the total system capacity is denoted by \(K\).

The application defines:

\[
K = s + \text{waiting-room capacity}
\]

where:

- \(s\) is the number of servers.
- The waiting-room capacity is the maximum number of customers allowed to wait.

When the system reaches state \(K\), new arrivals are blocked.

Therefore:

\[
\lambda_K = 0
\]

## Project Structure

```text
Markovian_Queues/
│
├── markovian_queues_app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── config.toml
