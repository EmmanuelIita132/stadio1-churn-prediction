"""Builds docs/literature_review.pdf (Part A) in the assignment's table format. Run: python docs/build_literature_review.py"""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

AUTHOR = "Emmanuel Iita (26304564)"   # <- edit before building
st = getSampleStyleSheet()
cell = ParagraphStyle("cell", parent=st["BodyText"], fontSize=9, leading=11.5, spaceAfter=2)
lab = ParagraphStyle("lab", parent=cell, fontName="Helvetica-Bold")
H2 = st["Heading2"]

def content(items):
    if isinstance(items, str):
        return [Paragraph(escape(items), cell)]
    return [Paragraph("&bull; " + escape(i), ParagraphStyle("b", parent=cell, leftIndent=10, firstLineIndent=-10)) for i in items]

def table(rows):
    data = [[Paragraph(escape(k), lab), content(v)] for k, v in rows]
    t = Table(data, colWidths=[3.4 * cm, 13.2 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t

pubs = [
 ("Publication 1", [
  ("Title", "Predicting Customer Churn: Extreme Gradient Boosting with Temporal Data"),
  ("Harvard reference", "Gregory, B. (2018) 'Predicting customer churn: extreme gradient boosting with temporal data', arXiv preprint arXiv:1802.03396. Available at: https://arxiv.org/abs/1802.03396 (Accessed: 28 September 2026)."),
  ("Link", "https://arxiv.org/abs/1802.03396"),
  ("Modelling", ["Primary model: XGBoost. Also tested LightGBM, StackNet and scikit-learn Gradient Boosting Machines.",
                 "Final submission: weighted-average ensemble of XGBoost (88%) and LightGBM (12%), with weights set using competition leaderboard feedback.",
                 "Hyperparameters tuned by cross-validation on the February 2017 period, using XGBoost's fast histogram tree grower.",
                 "Feature selection: each of 208 engineered features was added one at a time and kept only if cross-validated accuracy improved (76 kept); highly correlated overlaps were removed by hand.",
                 "Evaluation metric: log loss. Result: 0.07974, first place out of 575 teams."]),
  ("Preprocessing techniques", ["Data loaded into SQL Server and cleaned with T-SQL views: outlier removal, null imputation, and conversion of integer dates to date types.",
                 "Churn labels regenerated for each period with the organiser's Scala script (old labels discarded).",
                 "Time-based split: January 2017 for training, February for cross-validation, March for testing.",
                 "'Relative refactoring' of dates into days relative to the start of the prediction window (e.g. days since last login), so features mean the same in every period; absolute dates used only for registration and birth date.",
                 "Meta features across tables and higher-order ratio features (e.g. days since last login divided by plan length)."]),
  ("Datasets used", "KKBox churn dataset from the WSDM Cup 2018 challenge (Kaggle): user activity logs, transactions and member data covering three years."),
  ("Key takeaways", ["Build every feature relative to a fixed prediction date and split by time, to avoid leakage. This is the approach for the STADIO 1 pipeline (see Decision D2 in the README).",
                 "The strongest predictors were trends in recent activity (e.g. last two weeks vs the previous month), days since last activity, and cancellation history. No single feature dominated, so I will engineer many engagement-trend features from streaming and app data.",
                 "Gradient-boosted trees with careful feature engineering beat more complex approaches, so XGBoost/LightGBM will be my main benchmark."])]),
 ("Publication 2", [
  ("Title", "Time-to-Event Prediction with Neural Networks and Cox Regression"),
  ("Harvard reference", "Kvamme, H., Borgan, \u00d8. and Scheel, I. (2019) 'Time-to-event prediction with neural networks and Cox regression', Journal of Machine Learning Research, 20(129), pp. 1-30. Available at: https://jmlr.org/papers/v20/18-424.html (Accessed: 28 September 2026)."),
  ("Link", "https://jmlr.org/papers/v20/18-424.html (preprint: https://arxiv.org/abs/1907.00825; code: https://github.com/havakv/pycox)"),
  ("Modelling", ["Proposes Cox-MLP (CC) and Cox-Time: Cox proportional hazards models with the linear predictor replaced by a neural network; Cox-Time also removes the proportional-hazards assumption. A case-control sampled loss lets training run in mini-batches on very large data.",
                 "Compared with a linear Cox baseline (Cox-SGD), Cox-MLP (DeepSurv-style), DeepHit and Random Survival Forests.",
                 "Networks: multilayer perceptrons with ReLU, batch normalisation and dropout, trained with AdamWR. On KKBox: 6 to 8 layers of 128 to 512 nodes, chosen by grid search.",
                 "Metrics: time-dependent concordance, integrated Brier score (IBS) and integrated binomial log-likelihood (IBLL), which account for censoring.",
                 "KKBox results: Cox-Time had the best IBS (0.107) and IBLL (-0.334) with concordance 0.861. DeepHit had the highest concordance (0.888) but the worst calibration. RSF was strong (0.870)."]),
  ("Preprocessing techniques", ["Churn defined as no re-subscription within 30 days of expiry. Survival time is measured from the start of each subscription, with non-churners censored; a customer who churns and returns is treated as a new customer with covariates describing the earlier history.",
                 "15 covariates (7 numeric, 8 categorical) from member and transaction data only; user logs were not used.",
                 "Numeric covariates log-transformed (except number of previous churns) and standardised; categoricals encoded with entity embeddings.",
                 "Split into training (1,786,333 rows), test (661,748) and validation (198,665) sets."]),
  ("Datasets used", "KKBox churn dataset (large case study), plus four survival datasets (SUPPORT, METABRIC, Rotterdam and GBSG, FLCHAIN) and simulated data."),
  ("Key takeaways", ["Churn can be framed as time-to-event, which handles censored customers and gives a survival curve per subscriber showing when risk peaks. This suits the 30 to 60 day early-warning window in the STADIO 1 problem statement, and I will use a survival model alongside the binary classifier.",
                 "Discrimination alone is not enough: DeepHit ranked best but was poorly calibrated. Because STADIO 1 will act on risk scores, I will report calibration (Brier score) as well as AUC or concordance.",
                 "Clustering the predicted survival curves (K-means, 10 clusters) segmented customers by churn behaviour, e.g. a group at high immediate risk and spikes at month-end. This links directly to the STADIO 1 aim of segmenting its fanbase."])]),
 ("Publication 3", [
  ("Title", "Customer churn prediction in telecom using machine learning in big data platform"),
  ("Harvard reference", "Ahmad, A.K., Jafar, A. and Aljoumaa, K. (2019) 'Customer churn prediction in telecom using machine learning in big data platform', Journal of Big Data, 6(1), Article 28. doi:10.1186/s40537-019-0191-6. Available at: https://doi.org/10.1186/s40537-019-0191-6 (Accessed: 28 September 2026)."),
  ("Link", "https://doi.org/10.1186/s40537-019-0191-6 (open access: https://link.springer.com/article/10.1186/s40537-019-0191-6)"),
  ("Modelling", ["Four tree-based classifiers: Decision Tree (depth 20, 398 nodes), Random Forest (200 trees), Gradient Boosted Machine (200 trees) and XGBoost (180 trees). XGBoost was best with AUC 93.3%, then GBM (90.89%), Random Forest (87.76%) and Decision Tree (83%).",
                 "70/30 train/test split; hyperparameters tuned with 10-fold cross-validation. Metric: AUC.",
                 "Class imbalance (about 5% churners) tested three ways: oversampling, undersampling and no rebalancing. XGBoost and GBM did best without rebalancing; Random Forest and Decision Tree did better with undersampling.",
                 "Adding social network features raised AUC from 84% to 93.3%.",
                 "Validated on a fresh dataset (AUC 89%) and in a field test: customers predicted to churn were split into an offered group and a not-offered group, and about 47% of the offered group were retained."]),
  ("Preprocessing techniques", ["Built on Spark and a Hadoop cluster. Raw records were aggregated per customer by day, week and month (average, count, sum, max, min), producing thousands of columns.",
                 "Removed identifier columns, constant and duplicate features, features with more than 70% missing values and records with more than 90% missing; Pearson correlation used to drop correlated features.",
                 "Missing categories replaced with 'Other' and missing numerics with the feature mean; only the 31 most frequent categories were kept per categorical feature.",
                 "Social network features (PageRank, SenderRank, degree, neighbour connectivity, clustering coefficient, Jaccard and cosine similarity) from the last 4 months of calls and messages.",
                 "Labelled sample of 5 million customers (300,000 churners), with churn predicted two months ahead of the event."]),
  ("Datasets used", "Proprietary SyriaTel (Syria) telecom data: nine months of customer, network, call-detail, complaints and handset records for about 10 million customers. Not publicly available."),
  ("Key takeaways", ["Predicting two months ahead and testing with an offered/not-offered group is a useful template for showing that the STADIO 1 model leads to retention, not just a good AUC.",
                 "How much history to use matters: about 6 months of history helped the statistical features, older months added nothing, and boosted trees coped with a 5% minority class without resampling. I will test several look-back windows and compare resampling against class weights.",
                 "Relational features added the largest gain. For STADIO 1 the equivalents, such as fantasy-league groups and shared ticketing accounts, are worth exploring if the client can supply them."])]),
]
dataset = [
  ("Name", "WSDM - KKBox's Churn Prediction Challenge (KKBox churn dataset)"),
  ("Harvard reference", "KKBOX (2018) WSDM - KKBox's Churn Prediction Challenge [Data set]. Kaggle. Available at: https://www.kaggle.com/c/kkbox-churn-prediction-challenge (Accessed: 28 September 2026)."),
  ("Link", "https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data (a free Kaggle account is required, and the competition rules must be accepted before downloading)"),
  ("Number of instances", ["Labelled users: train.csv = 992,931 (subscriptions expiring Feb 2017); train_v2.csv = 970,960 (expiring Mar 2017, refreshed release).",
                 "Unlabelled competition test IDs: sample_submission_v2.csv = 907,471 (true labels are not public, so the labelled files must be split by time).",
                 "Supporting tables: members_v3.csv = 6,769,473 rows; transactions_v2.csv = 1,431,009 rows; user_logs_v2.csv = 18,396,362 rows. The original user_logs.csv is far larger (reported at about 392 million rows, roughly 30 GB); transactions.csv and user_logs.csv are the earlier releases.",
                 "Multiple splits: v1 (up to 28 Feb 2017) and v2 (up to 31 Mar 2017)."]),
  ("Number of feature columns", ["21 original feature columns across three tables, plus the user identifier msno (key) and the target is_churn.",
                 "members: city, bd (age), gender, registered_via, registration_init_time (5).",
                 "transactions: payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, is_auto_renew, transaction_date, membership_expire_date, is_cancel (8).",
                 "user_logs: date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs (8)."]),
  ("Number of classes (if applicable)", "2: is_churn = 1 (churned: no new valid subscription within 30 days after expiry) and is_churn = 0 (renewed). Churners are the minority class, so the data is imbalanced."),
  ("Target range (if applicable)", "Not applicable, as this is classification (binary target {0, 1}). For the survival-analysis extension, the target becomes a duration and an event indicator derived from the transactions, as in Kvamme et al. (2019)."),
  ("Origin", ["Real (not synthetic) data donated by KKBOX, a subscription music-streaming service based in Taiwan, for the WSDM Cup 2018 challenge, hosted on Kaggle. User IDs are anonymised.",
                 "Training data covers users whose membership expires in a given month (Feb 2017 for v1, Mar 2017 for v2); train and test are split by transaction date.",
                 "Prices are in New Taiwan Dollars.",
                 "It closely mirrors the STADIO 1 data request: subscription transactions, daily viewing or listening logs and member profiles from a paid streaming service, with the same 30-day non-renewal style of churn definition."]),
]
story = [Paragraph("Part A: Literature Review - Streaming Subscriber Churn Prediction", st["Title"]),
         Paragraph(escape(f"Project: STADIO 1 early-warning model for STADIO 1 TV subscriber churn. Student: {AUTHOR}"), cell),
         Paragraph("Three publications are summarised below, followed by the one public dataset to be used in Part B.", cell), Spacer(1, 8)]
for i, (name, rows) in enumerate(pubs):
    story += [Paragraph(name, H2), table(rows)]
    story.append(PageBreak())
story += [Paragraph("Dataset for Part B", H2), table(dataset)]
out = Path(__file__).with_name("literature_review.pdf")
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                  title="Part A literature review").build(story)
print("wrote", out)
