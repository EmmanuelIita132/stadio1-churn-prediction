# STADIO 1: Predicting Streaming Subscriber Churn

**Client:** STADIO 1 (commercial rights holder of the world's premier open-wheel motor-racing championship)
**Project:** Early-warning model to identify STADIO 1 TV subscribers likely to cancel
**Student:** Emmanuel Iita (26304564)

---

## Contents

1. [Motivation (Part A)](#1-motivation)
2. [Problem statement (Part B)](#2-problem-statement)
3. [Data request (Part C)](#3-data-request)
4. [Repository structure (Part D)](#4-repository-structure)
5. [RAAIDD log (Part E)](#5-raaidd-log)

---

## 1. Motivation

### Background
STADIO 1 has moved from selling its sport to broadcasters to speaking directly to fans through its streaming service, app and fantasy game. The Information Pack describes the direct-to-fan pillar as the smallest revenue line (10% of a USD 3.2 bn group revenue, roughly USD 320 m) but the fastest-growing, and the place "where the future margin is expected to come from". The 2030 Strategy aims to double this revenue.

### The problem this project addresses
The pack shows that the streaming subscriber base is leaking. Annual churn among STADIO 1 TV subscribers has risen from **19% to 26%** in two years. On the 3.4 million current subscribers, that seven-point rise is roughly **238,000 additional cancellations a year** (an illustrative estimate, since the pack does not give the year-by-year base). This is happening while free-app-to-paid conversion is also falling (**4.9% to 3.6%**), so STADIO 1 is losing fans at the exit faster than it wins them at the entrance. Average app engagement per race has also dropped from 38 to 31 minutes, which suggests that disengagement is visible in behaviour before a fan cancels.

### Why this matters now
- **It is an explicit board priority.** "Keep the subscribers we win" is one of five priorities in the 2030 Strategy. The pack asks for churn to be brought down "by spotting the fans drifting toward the exit before they cancel", which is the prediction task proposed here.
- **Newer fans are the most fickle.** The pack notes that fans won during the boom "churn faster and engage less deeply than the traditional core", so growth alone will not fix the problem. Retention is a cheaper lever than acquisition.
- **Retention currently uses no segmentation.** The pack states that all subscribers receive "the same generic emails and the same offers, at the same time". A churn-risk score allows retention effort to be targeted at the fans most likely to leave, instead of spread evenly.
- **The data already exists.** STADIO 1 holds 3 years of streaming history and 4 years of app activity, 5 years of ticketing and 3 years of email data. The pack also notes that these sit in silos and that fans appear under different identities across systems, so a model built on the joined-up view creates value beyond the prediction itself.

### Expected value to STADIO 1
1. **Lower churn and higher subscriber lifetime value**, through targeted retention offers to high-risk subscribers.
2. **More efficient marketing spend**, by replacing blanket offers with risk-based targeting and avoiding discounts for fans who would have stayed anyway.
3. **Insight into why fans leave.** The drivers of churn (for example lapsed engagement after a specific point in the season, or ticket no-shows) feed directly into the other 2030 priorities, including segmentation and conversion.
4. **A foundation for the fan-data platform.** The identity matching and feature engineering needed for this project support the fan-data platform described in Section 6 of the pack.

---

## 2. Problem statement

**Context.** STADIO 1's streaming annual churn has risen from 19% to 26% over two years across a base of 3.4 million subscribers. Retention actions are currently untargeted, and churn is only observed after the fan has cancelled.

**Problem.** STADIO 1 cannot identify which subscribers are likely to cancel in time to intervene.

**Question.** *Can we predict, using behavioural, transactional and engagement data from streaming, app and fantasy, ticketing, and email sources, which STADIO 1 TV subscribers will cancel or fail to renew in the next renewal period, early enough (for example 30 to 60 days ahead) for a retention action to change the outcome?*

**Type of task.** Supervised binary classification (target: churned vs retained at the next renewal), with a ranked churn-risk score per subscriber as the main output.

**Scope.**
- *In scope:* subscribers of STADIO 1 TV, with history from the last 3 years; features built from streaming, app and fantasy, ticketing, email and (where available) payment data; identification of the main drivers of churn.
- *Out of scope:* designing the retention offers themselves, sponsor analysis, ticket pricing and free-to-paid conversion (these are separate projects under the 2030 Strategy).

**Proposed evaluation.** A time-based train/validation/test split (train on earlier seasons, test on the latest) to avoid leakage. Because churners are a minority class (about 1 in 4), the model will be judged on precision-recall AUC, recall and lift in the top-risk deciles, and compared against a logistic regression baseline and a simple rule such as "no viewing in the last 30 days". Success thresholds will be agreed with the client once data is received.

**Stakeholders.** Direct-to-fan and digital product leadership, CRM and marketing teams, and data and technology teams responsible for the fan-data platform.

---

## 3. Data request

The full data request (Part C) is in [`docs/data_request.pdf`](docs/data_request.pdf). It covers subscriptions, billing, viewing sessions, the race calendar, app and fantasy activity, ticketing, email and marketing, fan profile, merchandise, and the identity mapping table.

---

## 4. Repository structure

```
stadio1-churn-prediction/
├── README.md                     <- this file (motivation, problem statement, structure, RAAIDD)
├── data/
│   ├── raw/                      <- Datasets: original data as received (never edited)
│   ├── interim/                  <- Datasets: cleaned and joined data
│   └── processed/                <- Datasets: final modelling tables and features
├── models/                       <- Models: saved trained models and their configuration
├── experiments/
│   ├── setup/                    <- Experimental setup: configs, splits, hyperparameter grids
│   └── results/                  <- Experimental results: metrics, predictions, logs
├── src/
│   ├── stats/                    <- Statistical helper and comparison scripts (e.g. model comparison tests)
│   ├── visualisation/            <- Visualisation scripts (EDA plots, ROC/PR curves, feature importance)
│   └── pipeline/                 <- Data cleaning, feature engineering and training code
├── notebooks/                    <- Exploratory notebooks
└── docs/
    └── data_request.pdf          <- Data request sent to the client (Part C)
```

| Required artefact | Location |
|---|---|
| Datasets | `data/` |
| Models | `models/` |
| Experimental setup | `experiments/setup/` |
| Experimental results | `experiments/results/` |
| Statistical helper and comparison scripts | `src/stats/` |
| Visualisation scripts | `src/visualisation/` |

---

## 5. RAAIDD log

### Risks

| ID | Risk | Likelihood / Impact | Mitigation |
|---|---|---|---|
| R1 | **Identity resolution fails.** The pack states the same fan can appear under different identities across app, streaming and ticketing. Poor matching would leave subscribers with incomplete feature sets or wrong app/ticketing history. | High / High | Request a master fan ID mapping table with a match-method and confidence column; measure the match rate; run the model with and without the low-confidence matches. |
| R2 | **Data leakage.** Features such as cancellation-flow activity, or viewing after the cancellation request, could reveal the outcome and inflate model performance. | Medium / High | Fix a prediction date per subscriber and build features only from data before it; use a time-based split. |
| R3 | **Seasonality distorts the labels.** Viewing follows the 24-race calendar, so off-season inactivity may look like churn risk, and the 3-year window covers only three seasons. | High / Medium | Build calendar-aware features (activity relative to race weekends); compare within season phase; test on a full held-out season. |
| R4 | **Client data arrives late, incomplete or sensitive.** Timestamps are inconsistent across time zones and the client is cautious about sharing data, which could delay the project. | Medium / High | Prove the approach on a public dataset first; request pseudonymised data; agree a fixed data-delivery date with the client. |
| R5 | **Class imbalance and a model with no business use.** With roughly 26% churn, accuracy would be a misleading metric, and a model without an actionable window is of little use. | Medium / Medium | Use PR-AUC and top-decile lift; define the prediction horizon (30 to 60 days) with the client. |

### Actions

| ID | Action | Output |
|---|---|---|
| A1 | Choose the client and scope the problem; write the motivation and problem statement | README (Parts A and B) |
| A2 | Submit the data request to STADIO 1 and agree the delivery format and date | `docs/data_request.pdf` |
| A3 | Create the GitHub repository and invite the lecturer | Repo with structure in Section 4 |
| A4 | Agree the definition of churn (voluntary cancellation vs non-renewal vs payment failure) and the prediction horizon | Written definition, logged as Decision D1 |
| A5 | Build the master fan ID table and join the streaming, app, ticketing and email data; standardise timestamps to UTC | `data/interim/` |
| A6 | Exploratory analysis of churn by tenure, plan type, acquisition cohort and engagement | Plots in `src/visualisation/`, notebooks |
| A7 | Engineer features (recency, frequency and trend of viewing and app use, race-weekend engagement, ticket no-shows, email responsiveness) | `data/processed/` |
| A8 | Train a baseline (logistic regression) and stronger models (e.g. gradient boosting); tune on the validation season | `models/`, `experiments/` |
| A9 | Evaluate on the held-out season; compare models statistically; interpret drivers | `experiments/results/`, `src/stats/` |
| A10 | Present results and recommendations to the client | Final report |

### Assumptions

| ID | Assumption | How it will be checked |
|---|---|---|
| AS1 | STADIO 1 records a reliable cancellation or non-renewal date for each subscriber, so a churn label can be built. | Confirm in the first data delivery; count records with missing dates. |
| AS2 | The last 3 years of streaming history is long enough to include subscribers who have gone through more than one renewal. | Check the tenure distribution and the number of renewals per subscriber. |
| AS3 | Engagement drops in the weeks before cancellation, so behavioural data has predictive signal. | Compare pre-cancellation engagement with retained subscribers in exploratory analysis. |
| AS4 | The client will provide pseudonymised (not personally identifiable) data, and consent covers use for modelling. | Confirm in writing with the client before data is transferred. |
| AS5 | The 19% to 26% churn figures in the pack use one consistent definition across both years. | Ask the client to confirm the definition (see Decision D1). |

### Issues

| ID | Issue | Status / Owner |
|---|---|---|
| I1 | No client data has been received yet, only the aggregate figures in the Information Pack, so feature availability and data quality cannot be checked. | Open. Data request issued (A2). |
| I2 | The pack does not define "churn" (voluntary cancellation, non-renewal or failed payment), and these have different causes and interventions. | Open. To be resolved through A4. |

### Decisions

| ID | Decision | Made by / Rationale |
|---|---|---|
| D1 | Project focus is **streaming churn prediction**, not conversion, ticket pricing, sponsor value or race excitement. | Student. It maps to a board priority, has a clear label, and uses data sources that hold 3 to 5 years of history. |
| D2 | Frame the task as **binary classification with a ranked risk score**, evaluated with a time-based split. | Student. Matches the business use (target the highest-risk fans) and avoids leakage. |
| D3 | Prove the approach on a **public dataset** before using client data. | Client and student. The client is cautious about sharing sensitive data. |

### Dependencies

| ID | Dependency | Reason |
|---|---|---|
| DEP1 | A1 must be completed before A2 | The data request depends on the agreed problem. |
| DEP2 | A2 (data delivered) must be completed before A5 | No joins or cleaning without the raw data. |
| DEP3 | A4 (churn definition) must be completed before A7 and A8 | Features need a fixed prediction date and label. |
| DEP4 | A5 (identity mapping) must be completed before A7 | Cross-source features require a reliable fan-level join. |
| DEP5 | A7 must be completed before A8, and A8 before A9 | Models require features; evaluation requires trained models. |
| DEP6 | A3 (repository) must be completed before any code or data is committed | All artefacts are stored in the repository. |

---

## 6. Assignment 2: proof of viability on the public KKBox dataset

As agreed in Decision D3, the approach is first proven on a public dataset: the WSDM KKBox Churn Prediction Challenge (a Taiwanese music-streaming service with subscriptions, daily usage logs and member profiles, and a 30-day non-renewal churn definition). Training cohort = February 2017 expiries; test cohort = March 2017 expiries (time-based split).

| Part | Document | Code |
|---|---|---|
| A: Literature review and dataset | [docs/literature_review.pdf](docs/literature_review.pdf) | n/a |
| B: Preprocessing | [Preprocessing.md](Preprocessing.md) | `src/pipeline/preprocessing.py` |
| B: Feature engineering | [FeatureEngineering.md](FeatureEngineering.md) | `src/pipeline/feature_engineering.py` |
| B: Model 1 (logistic regression) | [Model1.md](Model1.md) | `src/pipeline/model1.py` |
| B: Model 2 (XGBoost) | [Model2.md](Model2.md) | `src/pipeline/model2.py` |
| C: Model 1 performance | [Model1Performance.md](Model1Performance.md) | `src/stats/evaluate.py model1` |
| C: Model 2 performance | [Model2Performance.md](Model2Performance.md) | `src/stats/evaluate.py model2` |
| C: Comparison | [Comparison.md](Comparison.md) | `src/stats/evaluate.py compare` |
| D: Client recommendations report | [reports/STADIO1_Recommendations_Report.pdf](reports/STADIO1_Recommendations_Report.pdf) | `src/stats/build_report.py` |

### Run order
```bash
pip install -r requirements.txt
# download the KKBox files from Kaggle into data/raw/ (see Preprocessing.md)
python src/pipeline/preprocessing.py
python src/pipeline/feature_engineering.py
python src/pipeline/model1.py train && python src/pipeline/model1.py predict
python src/pipeline/model2.py train && python src/pipeline/model2.py predict
python src/stats/evaluate.py all
python src/stats/build_report.py "Your Name (student number)"
```
`src/pipeline/make_synthetic_data.py` creates a tiny fake dataset in the KKBox format for smoke-testing only; never report results from it.
