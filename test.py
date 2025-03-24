import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


from sklearn import set_config
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

import skfolio
import joblib

from skfolio import RatioMeasure, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.distribution import VineCopula
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import (
    DenoiseCovariance,
    DetoneCovariance,
    EWMu,
    GerberCovariance,
    ShrunkMu,
)
from skfolio.optimization import (
    MeanRisk,
    NestedClustersOptimization,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.pre_selection import SelectKExtremes
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import BlackLitterman, EmpiricalPrior, FactorModel, SyntheticData
from skfolio.uncertainty_set import BootstrapMuUncertaintySet



# === Fonction pour afficher les graphiques ===
def plot_portfolio_summary(portfolio, returns_index, start_date, end_date):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # === Données filtrées sur la période ===
    portfolio_returns = pd.Series(portfolio.returns, index=returns_index)
    portfolio_returns = portfolio_returns[start_date:end_date]

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    sharpe_ratio = portfolio.annualized_sharpe_ratio

    # === Graphique rendement cumulé ===
    plt.figure(figsize=(10, 5))
    cumulative.plot(label="Cumulé")
    plt.title("Rendement cumulé du portefeuille")
    plt.xlabel("Date")
    plt.ylabel("Valeur cumulée")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Graphique drawdown ===
    plt.figure(figsize=(10, 5))
    drawdown.plot(color="red", label=f"Drawdown (max: {max_drawdown:.2%})")
    plt.title("Drawdown du portefeuille")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Calculs de performance ===
    total_return = cumulative.iloc[-1] - 1
    nb_years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / nb_years) - 1
    annual_sharpe = portfolio.annualized_sharpe_ratio

    # === Détail des actifs investis ===
    weights = pd.Series(portfolio.weights, index=portfolio.assets)
    selected_assets = weights[weights > 0]
    nb_assets = len(selected_assets)

    # Classification (à adapter selon ton univers d’actifs)
    asset_types = {}
    for asset in selected_assets.index:
        if asset in ["IEF", "TLT", "BND", "LQD"]:
            asset_types[asset] = "Bond"
        elif asset in ["SPY", "AAPL", "MSFT", "GOOG", "JPM"]:
            asset_types[asset] = "Equity"
        else:
            asset_types[asset] = "Other"

    type_counts = Counter(asset_types.values())

    # === Création du tableau matplotlib ===
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table_data = [
        ["Période", f"{start_date} → {end_date}"],
        ["Nb jours", f"{len(portfolio_returns)}"],
        ["Rendement total", f"{total_return:.2%}"],
        ["Rendement annualisé", f"{annual_return:.2%}"],
        ["Sharpe annualisé", f"{annual_sharpe:.2f}"],
        ["Max drawdown", f"{max_drawdown:.2%}"],
        ["Nb actifs sélectionnés", f"{nb_assets}"],
    ]

    for type_name, count in type_counts.items():
        table_data.append([f"• {type_name}", f"{count} actif(s)"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Statistique", "Valeur"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Résumé de la performance du portefeuille", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # === Tableau DataFrame des actifs sélectionnés ===
    df_summary = pd.DataFrame({
        "Poids": selected_assets,
        "Type": [asset_types[a] for a in selected_assets.index]
    }).sort_values(by="Poids", ascending=False)

    print("\n📋 Détail des actifs sélectionnés dans le portefeuille :")
    print(df_summary.round(4))

    # === Pie chart de la répartition des poids par actif ===
    plt.figure(figsize=(7, 7))
    plt.pie(
        selected_assets.values,
        labels=selected_assets.index,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Répartition des poids par action dans le portefeuille")
    plt.axis('equal')  # Garder un cercle
    plt.tight_layout()
    plt.show()


    return df_summary  # Optionnel si tu veux l’exploiter plus tard






prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

model = MeanRisk()

model.fit(X_train)

print(model.weights_)

portfolio = model.predict(X_test)

print(portfolio.annualized_sharpe_ratio)
print(portfolio.summary())

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.SEMI_VARIANCE,
)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=EmpiricalPrior(
        mu_estimator=ShrunkMu(), covariance_estimator=DenoiseCovariance()
    ),
)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(),
)


model = MeanRisk(
    min_weights={"AAPL": 0.10, "JPM": 0.05},
    max_weights=0.8,
    transaction_costs={"AAPL": 0.0001, "RRC": 0.0002},
    groups=[
        ["Equity"] * 3 + ["Fund"] * 5 + ["Bond"] * 12,
        ["US"] * 2 + ["Europe"] * 8 + ["Japan"] * 10,
    ],
    linear_constraints=[
        "Equity <= 0.5 * Bond",
        "US >= 0.1",
        "Europe >= 0.5 * Fund",
        "Japan <= 1",
    ],
)
model.fit(X_train)

model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)

model = RiskBudgeting(
    prior_estimator=EmpiricalPrior(covariance_estimator=GerberCovariance())
)

model = NestedClustersOptimization(
    inner_estimator=MeanRisk(risk_measure=RiskMeasure.CVAR),
    outer_estimator=RiskBudgeting(risk_measure=RiskMeasure.VARIANCE),
    cv=KFold(),
    n_jobs=-1,
)


randomized_search = RandomizedSearchCV(
    estimator=MeanRisk(),
    cv=WalkForward(train_size=252, test_size=60),
    param_distributions={
        "l2_coef": loguniform(1e-3, 1e-1),
    },
)
randomized_search.fit(X_train)

best_model = randomized_search.best_estimator_

print(best_model.weights_)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=EmpiricalPrior(mu_estimator=EWMu(alpha=0.2)),
)

print(model.get_params(deep=True))

gs = GridSearchCV(
    estimator=model,
    cv=KFold(n_splits=5, shuffle=False),
    n_jobs=-1,
    param_grid={
        "risk_measure": [
            RiskMeasure.VARIANCE,
            RiskMeasure.CVAR,
            RiskMeasure.VARIANCE.CDAR,
        ],
        "prior_estimator__mu_estimator__alpha": [0.05, 0.1, 0.2, 0.5],
    },
)
gs.fit(X)

best_model = gs.best_estimator_

print(best_model.weights_)

views = ["AAPL - BBY == 0.03 ", "CVX - KO == 0.04", "MSFT == 0.06 "]
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=BlackLitterman(views=views),
)

factor_prices = load_factors_dataset()

X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

model = MeanRisk(prior_estimator=FactorModel())
model.fit(X_train, y_train)

print(model.weights_)

portfolio = model.predict(X_test)

print(portfolio.calmar_ratio)
print(portfolio.summary())

model = MeanRisk(
    prior_estimator=FactorModel(
        factor_prior_estimator=EmpiricalPrior(covariance_estimator=DetoneCovariance())
    )
)

factor_views = ["MTUM - QUAL == 0.03 ", "SIZE - TLT == 0.04", "VLUE == 0.06"]
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        factor_prior_estimator=BlackLitterman(views=factor_views),
    ),
)

set_config(transform_output="pandas")
model = Pipeline(
    [
        ("pre_selection", SelectKExtremes(k=10, highest=True)),
        ("optimization", MeanRisk()),
    ]
)
model.fit(X_train)

portfolio = model.predict(X_test)

model = MeanRisk()
mmp = cross_val_predict(model, X_test, cv=KFold(n_splits=5))
# mmp is the predicted MultiPeriodPortfolio object composed of 5 Portfolios (1 per testing fold)

mmp.plot_cumulative_returns()
print(mmp.summary())

model = MeanRisk()

cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2)

print(cv.summary(X_train))

population = cross_val_predict(model, X_train, cv=cv)

population.plot_distribution(
    measure_list=[RatioMeasure.SHARPE_RATIO, RatioMeasure.SORTINO_RATIO]
)
population.plot_cumulative_returns()
print(population.summary())


vine = VineCopula(log_transform=True, n_jobs=-1)
prior = SyntheticData(distribution_estimator=vine, n_samples=2000)
model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior)
model.fit(X)
print(model.weights_)

vine = VineCopula(log_transform=True, central_assets=["BAC"], n_jobs=-1)
vine.fit(X)
X_stressed = vine.sample(n_samples=10000, conditioning = {"BAC": -0.2})
ptf_stressed = model.predict(X_stressed)

vine = VineCopula(central_assets=["QUAL"], log_transform=True, n_jobs=-1)
factor_prior = SyntheticData(
    distribution_estimator=vine,
    n_samples=10000,
    sample_args=dict(conditioning={"QUAL": -0.2}),
)
factor_model = FactorModel(factor_prior_estimator=factor_prior)
model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)
model.fit(X, y)
print(model.weights_)

factor_model.set_params(factor_prior_estimator__sample_args=dict(
    conditioning={"QUAL": -0.5}
))
factor_model.fit(X,y)
stressed_X = factor_model.prior_model_.returns
stressed_ptf = model.predict(stressed_X)

#joblib.dump(model, "mymodel.pkl")  # vous pouvez nommer le fichier comme vous voulez

plot_portfolio_summary(portfolio, returns_index=X_test.index, start_date="2020-03-29", end_date="2022-12-31")