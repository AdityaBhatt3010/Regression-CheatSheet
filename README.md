# 📘 Regression CheatSheet — Machine Learning Regression Algorithms (with SHAP Explainability)

A clean, structured collection of **regression algorithms**, **code templates**, and **explainability tools** written in Python using scikit-learn, XGBoost, and SHAP.

This repository helps you:

- Understand **how each regression algorithm works**
- Learn **how to train, evaluate, and interpret** models
- Access **ready-to-run code templates** for real datasets or your own dataset
- Use **SHAP** to explain predictions of any regression model
- Explore models interactively using Jupyter notebooks

![Cover](https://github.com/user-attachments/assets/0a472a60-2f61-4eb2-af8f-9a2e2d11a34c) <br/>

---

## 🔧 Folder Structure

```

├── regression_cheatsheet.py         # Main script running ALL algorithms + SHAP summaries
├── regression_cheatsheet.ipynb      # Notebook version for interactive demonstration
├── SHAP_integration.py              # Explainability-specific file (for reference)
├── SHAP_integration.ipynb           # Fully working SHAP explainability notebook
├── requirements.txt                 # Dependencies
└── data/ (optional)
        └── your_dataset.csv         # Users may place their own CSV here

````

> **Note:**  
> SHAP visualizations require a **Jupyter environment**.  
> `SHAP_integration.py` is provided only for reference;  
> for working SHAP visualizations, use **`SHAP_integration.ipynb`**.

---

# 🚀 Getting Started

### Install dependencies:

Inside a Jupyter notebook:

```python
!pip install -r requirements.txt -q
````

Or from terminal:

```bash
pip install -r requirements.txt
```

### Run the complete cheat sheet:

```bash
python regression_cheatsheet.py
```

### Interactive notebook version:

```
regression_cheatsheet.ipynb
```

---

# 📊 Dataset Usage

The cheat sheet supports:

### **1. Diabetes dataset (default)**

Loaded from scikit-learn.

### **2. Custom user dataset**

If using your own dataset, place it inside:

```
data/your_file.csv
```

Then call:

```python
run_all(dataset="house")
```

Your CSV must contain:

* **3 input columns** → e.g., `column1`, `column2`, `column3`
* **1 target column** → e.g., `column4`

Example usage inside the code:

```python
df = pd.read_csv("data/filename.csv")
X = df[["column1", "column2", "column3"]].values
y = df["column4"].values
```

This repository does **not** include a dataset.
Users may supply any properly formatted CSV.

---

# 🧠 Algorithms Covered (Brief Theory + Code Snippets)

Below is a clear explanation of each algorithm, when to use it, and a short runnable snippet.

# 1️⃣ **Linear Regression**

A simple model assuming a linear relationship between features and target.

**Good for:** <br/>
✔ Simple relationships <br/>
✔ Quick baseline <br/>
✔ Interpretable coefficients <br/>

**Code:**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
```

---

# 2️⃣ **Ridge, Lasso, ElasticNet (Regularization)**

### 🔹 Ridge Regression

Adds **L2 penalty** → shrinks coefficients, reduces variance.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0).fit(X_train, y_train)
```

### 🔹 Lasso Regression

Adds **L1 penalty** → performs feature selection.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01).fit(X_train, y_train)
```

### 🔹 ElasticNet

Combination of L1 + L2.

```python
ElasticNet(alpha=0.01, l1_ratio=0.5)
```

---

# 3️⃣ **Polynomial Regression**

Transforms features into higher-degree polynomial combinations.

**Useful for:** <br/>
✔ Non-linear relationships <br/>
✔ Smooth curves <br/>

**Code:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("ridge", Ridge())
])
pipe.fit(X_train, y_train)
```

---

# 4️⃣ **Support Vector Regression (SVR)**

A powerful model using kernel functions to capture complex non-linear patterns.

**Pros:** <br/>
✔ Works well on small datasets <br/>
✔ Handles outliers using ε-insensitive loss <br/>

**Code:**

```python
from sklearn.svm import SVR
model = SVR(kernel="rbf").fit(X_train, y_train)
```

---

# 5️⃣ **Decision Tree Regression**

Learns decision boundaries in feature space.

**Pros:** <br/>
✔ Interpretable <br/>
✔ Handles non-linearity <br/>
✔ No scaling required <br/>

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor().fit(X_train, y_train)
```

---

# 6️⃣ **Random Forest Regression**

An ensemble of many decision trees.

**Pros:** <br/>
✔ High accuracy <br/>
✔ Low overfitting <br/>
✔ Handles noisy data <br/>

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor().fit(X_train, y_train)
```

---

# 7️⃣ **Gradient Boosting Regression**

Sequential ensemble of decision trees trained to correct previous errors.

**Pros:** <br/>
✔ Very accurate <br/>
✔ Works well on structured/tabular data <br/>
✔ Handles non-linearity <br/>

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor().fit(X_train, y_train)
```

---

# 8️⃣ **XGBoost Regression**

Extreme Gradient Boosting — optimized, regularized, fast.

**Pros:** <br/>
🔥 State-of-the-art performance on tabular data <br/>
🔥 Built-in regularization <br/>
🔥 GPU acceleration <br/>

```python
import xgboost as xgb
model = xgb.XGBRegressor().fit(X_train, y_train)
```

---

# 9️⃣ **Pipeline + Ridge Regression**

Combining preprocessing + model into a single workflow.

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])
pipe.fit(X_train, y_train)
```

---

# 📈 Evaluation Metrics Used

| Metric                        | Measures                    | Notes                 |
| ----------------------------- | --------------------------- | --------------------- |
| **R² Score**                  | Variance explained by model | Higher → better       |
| **MSE (Mean Squared Error)**  | Squared error               | Sensitive to outliers |
| **MAE (Mean Absolute Error)** | Absolute error              | More robust           |

Example:

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

---

# 🔍 SHAP Explainability

Every model in the cheat sheet runs SHAP explainability.

### What SHAP provides:

* Feature importance
* Beeswarm summary plots
* Local prediction explanations

### Example:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

### Important:

> **SHAP visualizations only work in Jupyter.**
> For working examples, use **`SHAP_integration.ipynb`**.
> `SHAP_integration.py` is for reference only.

---

# 🏃 Running Everything at Once

```bash
python regression_cheatsheet.py
```

This runs:

* Linear Regression
* Ridge
* Lasso
* ElasticNet
* Polynomial Regression
* SVR
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* SHAP Explainability for all models

---

# ⭐ Future Extensions (optional)

* Add LightGBM & CatBoost variants
* Add hyperparameter tuning (GridSearch / Optuna)
* Add Regression Comparison Dashboard
* Add more datasets (Salary, Cars, Custom Synthetic Data)

---

### 👋 **Goodbye Note**

Good luck, and may your R² rise, your MSE fall,
and your SHAP plots always make sense.

---

### 🐾 Follow Me

If you enjoyed this analysis, check out more of my work:

🔗 [GitHub](https://github.com/AdityaBhatt3010) <br/>
💼 [LinkedIn](https://www.linkedin.com/in/adityabhatt3010/) <br/>
✍️ [Medium](https://medium.com/@adityabhatt3010) <br/>

---

# 👨‍💻 Crafted By  

**Aditya Bhatt** — Turning black-box models into transparent systems.

---




