"""Part D: build the client recommendations report (Markdown + PDF) from the REAL results.

Usage (after `python src/stats/evaluate.py all`):
    python src/stats/build_report.py "Your Name (student number)"
Output: reports/STADIO1_Recommendations_Report.md and .pdf
All numbers and the winner/loser wording come from experiments/results/*.json, so the report always matches the results.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

author = sys.argv[1] if len(sys.argv) > 1 else "Emmanuel Iita (26304564)"
R = json.load(open(C.RESULTS / "comparison.json"))
cfg2 = json.load(open(C.SETUP / "model2_config.json"))
win = R["winner"]; lose = "Model 1" if win == "Model 2" else "Model 2"
name = {"Model 1": "logistic regression (Model 1)", "Model 2": "XGBoost (Model 2)"}
mw, ml = R["model2" if win == "Model 2" else "model1"], R["model1" if win == "Model 2" else "model2"]
pb, sig = R["paired_bootstrap"], R["significant"]
imp = pd.read_csv(C.RESULTS / "model2_feature_importance.csv", index_col=0).iloc[:, 0]
top3 = ", ".join(f"'{i}'" for i in imp.index[:3])
f3 = lambda v: f"{v:.3f}"
churn_rate = json.load(open(C.RESULTS / "model1_metrics.json"))["churn_rate"]
sigtxt = ("The paired bootstrap shows the gap is statistically significant at the 5% level"
          if sig else "The paired bootstrap does not show a statistically significant gap on both key metrics, so the choice rests on secondary criteria")
aligned = (win == "Model 2") and sig
align_txt = (
    "Our findings are consistent with the three studies reviewed. Gregory (2018) won the KKBox challenge with gradient-boosted trees, and "
    "Ahmad, Jafar and Aljoumaa (2019) also found XGBoost the strongest of four tree-based classifiers; our result, in which XGBoost outperforms a "
    "linear baseline, agrees with both. Trends in recent activity and days since last activity ranked among Gregory's strongest predictors, and our "
    f"most important features ({top3}) point the same way, although our list is dominated by subscription-state variables. Kvamme, Borgan and Scheel (2019) "
    "warn that discrimination alone is insufficient, so we also report calibration; our models are evaluated on log loss, Brier score and expected calibration error as well as AUC."
    if aligned else
    "Our findings only partly agree with the literature. Gregory (2018) and Ahmad, Jafar and Aljoumaa (2019) both found gradient-boosted trees superior "
    "to simpler models, but in our experiment " + ("XGBoost's advantage was not statistically significant. " if win == "Model 2" else "the linear baseline performed at least as well. ") +
    "Likely reasons are our smaller sampled data set, a limited hyper-parameter search and a shorter feature set than the 208 engineered features used by Gregory. "
    "Consistent with Kvamme, Borgan and Scheel (2019), we report calibration alongside discrimination.")

S = []  # (kind, text)
S.append(("title", "Recommendations on Predicting STADIO 1 TV Subscriber Churn"))
S.append(("meta", f"Prepared for: STADIO 1 Direct-to-Fan and Digital Product leadership | Prepared by: {author} | 28 September 2026"))
S.append(("h", "Purpose"))
S.append(("p", f"STADIO 1 asked whether subscriber churn can be predicted early enough to act on. Before using sensitive client data, we tested our approach on the public KKBox music-streaming data set, which closely mirrors STADIO 1's subscription, billing and usage data (KKBOX, 2018). Two models were trained on February 2017 renewals and tested on the following month's renewals, so the test reflects genuine forecasting. The test period contained {churn_rate:.1%} churners. This report answers four questions."))
S.append(("h", "1. Which model is best?"))
S.append(("p", f"We recommend {name[win]}. It achieved a log loss of {f3(mw['log_loss'])} against {f3(ml['log_loss'])}, and a precision-recall AUC of {f3(mw['pr_auc'])} against {f3(ml['pr_auc'])} (ROC AUC {f3(mw['roc_auc'])} against {f3(ml['roc_auc'])}). {sigtxt}."))
calib_txt = ("the scores can be read as approximate probabilities, which matters when deciding how much to spend on an offer"
             if mw["ece"] < 0.02 else "the scores should be recalibrated before being read as probabilities when sizing an offer")
S.append(("p", f"For retention teams the practical measure is targeting efficiency. Contacting only the 10% of subscribers with the highest risk scores would reach {mw['top10_recall']:.0%} of all churners, a lift of {mw['top10_lift']:.1f} times random targeting. Calibration error was {f3(mw['ece'])}, so {calib_txt}. At the chosen threshold the model reached a precision of {mw['precision']:.0%} and a recall of {mw['recall']:.0%}, against {R['rule']['precision']:.0%} and {R['rule']['recall']:.0%} for a simple rule that flags subscribers whose auto-renew is switched off."))
S.append(("p", f"The two models differ in more than accuracy. Logistic regression is easy to explain to CRM and marketing teams, but it can only capture linear effects. Gradient-boosted trees such as XGBoost (Chen and Guestrin, 2016) capture interactions, for example a subscriber whose viewing is falling and who also has auto-renew switched off. In our XGBoost model the most influential features were {top3}, which gives the business a first view of why fans leave, although importance scores describe association rather than cause. Both models were tuned on a validation slice only, and the test period was never used for tuning, so the figures above are an honest estimate of forward performance."))
S.append(("h", "2. How can the selected model be improved?"))
S.append(("p", "We propose five steps, in order of expected value. First, widen the search: tune tree depth, learning rate and regularisation with Bayesian optimisation over several time-based folds, rather than the three-value depth grid used here. Second, enrich the features, testing alternative look-back windows and further engagement-trend measures, since Gregory (2018) found trends in recent activity to be among the strongest signals. Third, blend the model with a second boosted-tree variant such as LightGBM, as the winning KKBox entry did. Fourth, add a survival formulation (Kvamme, Borgan and Scheel, 2019), which produces a risk curve per subscriber and therefore suggests when to intervene, not only whom. Fifth, replace the F1-based threshold with a cost-based one that reflects the price of an offer against the lifetime value of a retained fan. Throughout, scores should be recalibrated periodically, and performance should be monitored monthly so that drift, caused for instance by a price change or a new plan, is detected before it erodes the value of the retention programme. Class weighting and resampling should also be tested against the current unweighted approach, because the churn share at STADIO 1 may differ from that of KKBox."))
S.append(("h", "3. How must the model be adapted to the STADIO 1 data?"))
S.append(("p", "The KKBox tables map directly onto our data request: transactions to Subscriptions and Billing (Tables 1 and 2), listening logs to Viewing sessions and App activity (Tables 3 and 5), and members to Fan profile (Table 9). Five adaptations are needed."))
S.append(("p", "(a) Churn definition. KKBox defines churn as no renewal within 30 days of expiry. STADIO 1's 19% to 26% figures are undefined, so we must agree whether voluntary cancellation, non-renewal and payment failure are modelled together or separately, using the cancellation type field. (b) Prediction date. KKBox uses one monthly cohort; STADIO 1 has monthly, annual and seasonal plans, so each subscriber needs a prediction date 30 to 60 days before their own renewal. (c) Seasonality. Viewing follows the 24-race calendar, so features must be measured relative to race weekends and the race characteristics in Table 4, and the model validated on a complete held-out season. (d) Identity. Features from the app, ticketing and email sources require the master fan identifier in Table 10; matches of low confidence should be tested with and without. (e) Governance. Data must be pseudonymised, timestamps standardised to UTC, and use covered by consent. Finally, we recommend a controlled pilot in which a random share of high-risk subscribers receives no offer, following the field test of Ahmad, Jafar and Aljoumaa (2019), so that the retention benefit can be measured. Retention offers themselves remain outside this model's scope."))
S.append(("h", "4. Do the results align with the literature?"))
S.append(("p", align_txt))
S.append(("p", "One caveat applies: KKBox is a music service with one plan type and no seasonality, so absolute performance should not be expected to transfer to STADIO 1. The value of this exercise is that the method, the leakage controls and the evaluation design have been proven and can now be applied to STADIO 1 data."))
S.append(("h", "References"))
for r in ["Ahmad, A.K., Jafar, A. and Aljoumaa, K. (2019) 'Customer churn prediction in telecom using machine learning in big data platform', Journal of Big Data, 6(1), Article 28. doi:10.1186/s40537-019-0191-6.",
          "Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting system', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794.",
          "Gregory, B. (2018) 'Predicting customer churn: extreme gradient boosting with temporal data', arXiv preprint arXiv:1802.03396. Available at: https://arxiv.org/abs/1802.03396 (Accessed: 28 September 2026).",
          "KKBOX (2018) WSDM - KKBox's Churn Prediction Challenge [Data set]. Kaggle. Available at: https://www.kaggle.com/c/kkbox-churn-prediction-challenge (Accessed: 28 September 2026).",
          "Kvamme, H., Borgan, O. and Scheel, I. (2019) 'Time-to-event prediction with neural networks and Cox regression', Journal of Machine Learning Research, 20(129), pp. 1-30."]:
    S.append(("ref", r))

out_dir = C.ROOT / "reports"; out_dir.mkdir(exist_ok=True)
md = "\n\n".join(("# " if k == "title" else "## " if k == "h" else "") + t for k, t in S)
(out_dir / "STADIO1_Recommendations_Report.md").write_text(md)
body_words = sum(len(t.split()) for k, t in S if k in ("p",))
print(f"body word count (excluding references): {body_words}")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape
st = getSampleStyleSheet()
sty = {"title": st["Title"], "h": st["Heading2"], "p": ParagraphStyle("p", parent=st["BodyText"], leading=14, spaceAfter=6),
       "meta": ParagraphStyle("m", parent=st["BodyText"], textColor="#555555", fontSize=9),
       "ref": ParagraphStyle("r", parent=st["BodyText"], fontSize=9, leftIndent=14, firstLineIndent=-14)}
doc = SimpleDocTemplate(str(out_dir / "STADIO1_Recommendations_Report.pdf"), pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                        topMargin=2*cm, bottomMargin=2*cm, title="STADIO 1 churn recommendations", author=author)
doc.build([Paragraph(escape(t), sty[k]) if k != "spacer" else Spacer(1, 6) for k, t in S])
print("wrote", out_dir / "STADIO1_Recommendations_Report.pdf")
