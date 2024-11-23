## ML FLow experiements

MLFLOW_TRACKING_URI=https://dagshub.com/krishnaik06/mlflowexperiments.mlflow \
MLFLOW_TRACKING_USERNAME=krishnaik06 \
MLFLOW_TRACKING_PASSWORD=7104284f1bb44ece21e0e2adb4e36a250ae3251f \
python script.py

### What is MLflow?

MLflow is an **open-source platform** designed to manage the **entire machine learning lifecycle**. It helps track experiments, package models, deploy them, and monitor their performance—all in one place. Think of MLflow as a **centralized tool** for handling everything you do in a machine learning project.

---

### Why is MLflow Needed?

Machine learning projects involve many steps like:
1. **Experimenting with different models** (e.g., Logistic Regression vs. Random Forest).
2. **Tuning hyperparameters** (e.g., learning rate, number of layers).
3. **Keeping track of code changes**.
4. **Packaging models** so they can be shared or deployed.
5. **Monitoring deployed models** to ensure they're still working well.

Managing all of this can quickly become messy without a proper tool. **MLflow organizes and simplifies these tasks**, making collaboration and experimentation more manageable.

---

### Core Features of MLflow

MLflow has **four main components**:

1. **Tracking**: 
   - Logs experiments, metrics (accuracy, loss), and parameters (learning rate, batch size).
   - Example: You try three different models and want to compare their performance. MLflow saves the details of each run for easy comparison.

2. **Projects**:
   - Packages your code in a reusable format.
   - Example: A colleague wants to run your model on their system. Instead of explaining how to set up the environment, you share an MLflow project, and it just works.

3. **Models**:
   - Saves your machine learning model in a standard format so it can be deployed anywhere.
   - Example: After training, you save the model in a way that works with web apps, APIs, or mobile apps.

4. **Model Registry**:
   - Manages the lifecycle of models, like tracking versions and keeping track of which model is in production.
   - Example: When deploying a new version of a model, MLflow ensures you can easily roll back if something goes wrong.

---

### How MLflow Works (Real-Time Example)

#### Scenario: Predicting Customer Churn for a Telecom Company

1. **Experiment Tracking**:
   - You try three models: Logistic Regression, Random Forest, and XGBoost.
   - For each model, you adjust hyperparameters (like max_depth for Random Forest).
   - MLflow logs:
     - The model used (e.g., Logistic Regression).
     - Parameters (e.g., `penalty='l2'`).
     - Metrics (e.g., accuracy = 85%, precision = 90%).
   - Later, you can compare runs in a dashboard and see which model performed best.

2. **Projects**:
   - You package your code (data preprocessing, model training) as an MLflow project.
   - A new data scientist joins the team. Instead of explaining how to set up dependencies, you share the MLflow project, and they can replicate your work with one command.

3. **Models**:
   - The best model (XGBoost) is saved using MLflow.
   - It's stored in a format that can be used by web developers to integrate into a web app for real-time predictions.

4. **Model Registry**:
   - The XGBoost model is marked as "Production" in the registry.
   - Later, if you train a new model (e.g., a deep learning model), it can be marked as "Staging" for testing before replacing the production model.

---

### Benefits of MLflow

1. **Organization**:
   - Keeps experiments, code, and models organized.
2. **Collaboration**:
   - Teams can easily share and compare work.
3. **Reproducibility**:
   - Logs everything to ensure experiments can be repeated.
4. **Simplified Deployment**:
   - Saves models in a format ready for production.

---

### Analogy

Think of MLflow as a **project manager** for machine learning:
- **Tracking** is like taking notes on every task.
- **Projects** are like pre-packed toolkits for others to use.
- **Models** are like ready-to-use products.
- **Model Registry** is like keeping a catalog of all products with their versions and status (e.g., in use or retired).

---
Great question! MLflow and Docker do have similarities, but they serve **different purposes**. Let’s break it down.

---

### Is MLflow Similar to Docker?

**Similarities:**
1. Both tools help with **reproducibility**:
   - **MLflow** ensures your machine learning experiments, models, and workflows are consistent across environments.
   - **Docker** ensures your entire application (not just ML) runs the same way everywhere by packaging it into a container with all its dependencies.

2. Both are used to **share and deploy** work:
   - **MLflow** focuses specifically on machine learning workflows (experiments, model tracking, deployments).
   - **Docker** can package and deploy any software, including ML models or APIs.

**Key Difference:**
- **MLflow** is **machine-learning-specific** and focuses on tracking experiments, storing models, and managing model lifecycles.
- **Docker** is a general-purpose **containerization tool** that runs applications in isolated environments, not specific to machine learning.

**Example**: 
You could use MLflow to:
1. Log the training process and save your ML model.
2. Package and deploy the MLflow model using Docker for a production environment.

In short, they often **complement each other** rather than replacing each other.

---

### Is MLflow Used in Real-Time by Everyone?

**1. In Real-Time?**
- Yes, MLflow is widely used in **real-world machine learning workflows**, especially in organizations with complex ML pipelines. 
- Companies like **Databricks, Microsoft, and Airbnb** use MLflow extensively for their machine learning lifecycle management.

**2. Is Everyone Using It?**
- **Not everyone.** MLflow is popular, but some teams might use alternative tools, like:
  - **Kubeflow**: For managing end-to-end ML workflows in Kubernetes.
  - **Weights & Biases (W&B)**: Focused more on experiment tracking and collaboration.
  - **TensorBoard**: Used with TensorFlow for tracking and visualization.
  - Custom-built solutions: Some teams build their own systems for their specific needs.

**3. Why Isn’t It Universal?**
- MLflow may not be suitable for:
  - Very small projects (where its features might feel like overkill).
  - Organizations already committed to a competing tool.
  - Teams that don’t yet have a mature machine learning pipeline.

---

### When Should You Use MLflow?

1. **Collaborative Projects**: If you’re working in a team, MLflow ensures that everyone can track and share progress.
2. **Experiment-Heavy Workflows**: If you’re testing many models or hyperparameters, MLflow keeps things organized.
3. **Production-Ready Models**: MLflow’s model registry makes deploying and managing production models easier.
4. **Scaling ML Systems**: Larger organizations with complex pipelines benefit greatly from MLflow.

---

### Real-Life Example of MLflow and Docker Together

#### Scenario: E-Commerce Product Recommendation System
1. **MLflow**:
   - Track experiments with different recommendation models.
   - Save the best-performing model (e.g., Collaborative Filtering).
   - Register the model as "Production" in the MLflow registry.

2. **Docker**:
   - Create a container with all dependencies (Python, MLflow, model).
   - Deploy the container to a cloud server (e.g., AWS, Azure).

Now, both MLflow and Docker ensure the system works consistently, from the data scientist's laptop to the production server.

---

In summary:
- MLflow is **not a replacement** for Docker but works well with it.
- It’s a **popular tool** in real-world applications but isn’t used by everyone. Alternatives like Kubeflow or W&B also have strong user bases.
---
### MLflow vs. Other Tools: A Comparison

To understand MLflow's position better, let’s compare it with some popular alternatives: **Kubeflow**, **Weights & Biases (W&B)**, and **TensorBoard**.

---

#### 1. **MLflow vs. Kubeflow**

| **Feature**               | **MLflow**                               | **Kubeflow**                          |
|----------------------------|------------------------------------------|---------------------------------------|
| **Purpose**                | Manages the ML lifecycle: experiment tracking, model packaging, deployment, and registry. | Focuses on orchestrating end-to-end ML workflows on Kubernetes. |
| **Ease of Use**            | Simpler, beginner-friendly, requires no Kubernetes expertise. | Complex, requires Kubernetes knowledge. |
| **Experiment Tracking**    | Built-in, user-friendly.                 | Supported but not as intuitive.       |
| **Deployment**             | Supports deployment with tools like Flask, Docker. | Strong integration with cloud-native environments. |
| **Best For**               | Small to medium-scale projects.          | Enterprise-grade, large-scale ML systems. |

**Example Use**:  
- MLflow: A small startup tracking experiments and deploying models on AWS.  
- Kubeflow: A large enterprise automating a machine learning pipeline across multiple cloud servers.

---

#### 2. **MLflow vs. Weights & Biases (W&B)**

| **Feature**               | **MLflow**                               | **Weights & Biases (W&B)**            |
|----------------------------|------------------------------------------|---------------------------------------|
| **Experiment Tracking**    | Logs parameters, metrics, and artifacts. | Provides an interactive dashboard with real-time collaboration. |
| **Collaboration**          | Basic (logs in a central place).         | Highly collaborative with team-focused features. |
| **Ease of Integration**    | Supports many frameworks (e.g., TensorFlow, PyTorch). | Similar integration support but tailored dashboards. |
| **Cost**                   | Free and open-source.                    | Free tier, but advanced features require payment. |
| **Best For**               | Teams needing a general-purpose ML tool. | Teams needing detailed experiment tracking and collaboration. |

**Example Use**:  
- MLflow: A solo data scientist tracking experiments and managing models.  
- W&B: A research team collaborating on cutting-edge AI experiments.

---

#### 3. **MLflow vs. TensorBoard**

| **Feature**               | **MLflow**                               | **TensorBoard**                       |
|----------------------------|------------------------------------------|---------------------------------------|
| **Experiment Tracking**    | Logs across any ML framework.            | Primarily designed for TensorFlow models. |
| **Visualization**          | Offers metrics comparison and graphs.    | Excellent for detailed graph visualizations (e.g., neural network layers). |
| **Model Deployment**       | Supports deployment with registries.     | Not focused on deployment.            |
| **Ease of Use**            | Easy setup for various frameworks.       | Best for TensorFlow users.            |
| **Best For**               | Diverse ML workflows.                    | TensorFlow-centric projects.          |

**Example Use**:  
- MLflow: A company using PyTorch for training and Flask for deployment.  
- TensorBoard: A team fine-tuning TensorFlow-based deep learning models.

---

### Real-Life Industries Using MLflow

#### 1. **Healthcare**
- **Use Case**: Predicting disease progression using patient data.
- **Example**: A hospital uses MLflow to track multiple machine learning models for predicting diabetes risk.  
  - **Why MLflow?** To compare models (e.g., logistic regression vs. neural networks) and deploy the best one to a production system.

#### 2. **E-commerce**
- **Use Case**: Personalizing recommendations.
- **Example**: An online retailer logs experiments with MLflow to improve product recommendation algorithms and deploy them to their website.

#### 3. **Finance**
- **Use Case**: Fraud detection.
- **Example**: A bank tracks models trained to detect fraudulent transactions using MLflow.  
  - **Why MLflow?** To monitor model accuracy and retrain models when they degrade.

#### 4. **Manufacturing**
- **Use Case**: Predictive maintenance.
- **Example**: A factory uses MLflow to log models predicting when equipment will need servicing.  
  - **Why MLflow?** To manage multiple versions of models across different machines.

---

### Summary: MLflow in the Real World

| **Tool**     | **Best Fit**                                                                                  |
|--------------|----------------------------------------------------------------------------------------------|
| **MLflow**   | General-purpose ML lifecycle management. Ideal for diverse ML projects with modest complexity. |
| **Kubeflow** | Large-scale, cloud-native ML pipelines requiring orchestration and automation.               |
| **W&B**      | Teams prioritizing collaboration and in-depth experiment tracking.                          |
| **TensorBoard** | TensorFlow-centric workflows needing visualization of neural networks.                    |

---
Here’s a detailed and simplified explanation of the provided MLflow code. Each block is broken down for clarity, along with examples to help you understand its purpose.

---

### **1. Importing Libraries**

```python
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import logging
```

- **Purpose**: Importing essential libraries for data processing, model building, evaluation, and MLflow.
- **Key Components**:
  - **Pandas**: Used for handling tabular data (like spreadsheets).
  - **Numpy**: For mathematical operations.
  - **Scikit-learn**: Provides machine learning algorithms and evaluation metrics.
  - **MLflow**: Tracks machine learning experiments, logs parameters, and manages models.
  - **Logging**: Helps track warnings or errors in the code.
  
**Example**: Think of these imports as packing all tools you'll need for a cooking recipe before starting.

---

### **2. Setting Up Logging**

```python
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```

- **Purpose**: Sets up a logging system to capture warnings or errors.
- **Why?** Logs are helpful in debugging or understanding what went wrong in the code.

**Example**: Like keeping a diary of events during a project to review later.

---

### **3. Evaluation Function**

```python
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
```

- **Purpose**: Calculates how good the model's predictions are using three metrics:
  - **RMSE (Root Mean Squared Error)**: Measures how far the predictions are from the actual values.
  - **MAE (Mean Absolute Error)**: Average of absolute errors.
  - **R2 Score**: How well the model explains the variability of the data.

**Example**: Like grading a student’s performance with percentages, letter grades, and class rank.

---

### **4. Reading the Dataset**

```python
csv_url = (
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
)
try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
```

- **Purpose**: Downloads and reads the wine-quality dataset.
- **Why?** MLflow tracks experiments on real-world data. This dataset contains wine characteristics (input) and wine quality scores (output).
- **Exception Handling**: If the data fails to load, an error message is logged.

**Example**: Imagine downloading a recipe from a website to test cooking skills. If the site is down, you log an error.

---

### **5. Splitting Data for Training and Testing**

```python
train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]
```

- **Purpose**: 
  - **train_test_split**: Divides the dataset into training (75%) and testing (25%) parts.
  - **train_x, test_x**: Input features (wine properties like acidity, pH).
  - **train_y, test_y**: Target values (wine quality score).
  
**Example**: Like practicing cooking (training) before serving a guest (testing).

---

### **6. Model Parameters**

This part of the code is a **conditional assignment** that determines the values of the variables `alpha` and `l1_ratio`. Let’s break it down:

---

### **Code Explanation**

```python
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
```

- **What It Does**:
  - Checks if a value is provided for `alpha` through the command line (via `sys.argv`).
  - If a value exists (i.e., the length of `sys.argv` is greater than 1), it takes that value (converted to a float).
  - Otherwise, it uses the default value `0.5`.

- **Key Components**:
  - `sys.argv`: A list that stores command-line arguments passed to the script.
    - `sys.argv[0]`: The script's name.
    - `sys.argv[1]`: The first argument passed to the script (if provided).
  - `len(sys.argv) > 1`: Checks if at least one additional argument is provided.
  - `float(sys.argv[1])`: Converts the first argument into a floating-point number.
  - `else 0.5`: Uses `0.5` as the default value if no argument is provided.

---

```python
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
```

- **What It Does**:
  - Similar to the `alpha` logic, this assigns a value to `l1_ratio` based on whether a second command-line argument is provided.
  - If a second argument exists, it uses that value (converted to a float).
  - Otherwise, it defaults to `0.5`.

---

### **Simplified Explanation**

This code is saying:

- *"If the user provides a value for `alpha` and `l1_ratio` while running the script, use those values. Otherwise, use `0.5` as the default for both."*

---

### **Real-Time Example**

Imagine you’re ordering a pizza:

1. If you call and specify a **pizza size** (e.g., *large*), the restaurant uses your choice.
2. If you don’t specify a size, they give you the **default size** (e.g., *medium*).

In this case:
- `sys.argv[1]` = The size you specify when ordering (*large*).
- Default = The size they assume if you don’t specify (*medium*).

---

### **How It Works When Running the Script**

1. **With Arguments:**
   ```bash
   python script.py 0.7 0.3
   ```
   - `sys.argv[1]` → `0.7` (used as `alpha`).
   - `sys.argv[2]` → `0.3` (used as `l1_ratio`).

   Result:
   ```python
   alpha = 0.7
   l1_ratio = 0.3
   ```

2. **Without Arguments:**
   ```bash
   python script.py
   ```
   - No additional arguments are provided.
   - Defaults are used:
     ```python
     alpha = 0.5
     l1_ratio = 0.5
     ```

---

### Why This is Useful

- It makes the script **flexible**:
  - You can quickly experiment with different values by passing them via the command line.
  - Default values (`0.5`) act as a fallback, so the script doesn’t break if no arguments are given.

### **7. Training the Model with MLflow**
---

### Code:
```python
with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
```

---

### Understanding the Model: **ElasticNet**

1. **What is ElasticNet?**
   - **ElasticNet** is a type of linear regression model. It combines the features of **Lasso Regression** (L1 regularization) and **Ridge Regression** (L2 regularization).
   - The `alpha` and `l1_ratio` parameters determine the balance between L1 and L2 regularization:
     - **`alpha`**: Controls the overall strength of regularization. Higher values mean stronger regularization.
     - **`l1_ratio`**: Controls the mix between L1 (Lasso) and L2 (Ridge). 
       - `l1_ratio=1.0`: Pure Lasso (only L1 regularization).
       - `l1_ratio=0.0`: Pure Ridge (only L2 regularization).
       - A value in between (e.g., `0.5`) mixes both.

2. **Why Use ElasticNet Instead of Simple Linear Regression?**
   - Regularization is applied to prevent **overfitting** by adding penalties for large coefficients.
   - ElasticNet is particularly useful when:
     - Your data has **many features**.
     - Some features are **correlated** or **irrelevant**.

3. **How Does This Relate to Linear Regression?**
   - ElasticNet builds on the **linear regression model**, where predictions are made by finding the line (or hyperplane) that minimizes the error.
   - In ElasticNet, an additional penalty (based on L1 and L2 norms) is added to the cost function to control complexity.

---

### Code Walkthrough

#### **1. Starting an MLflow Run**
```python
with mlflow.start_run():
```
- This block tells MLflow to start tracking the training process, allowing you to log parameters, metrics, and the model.

---

#### **2. Defining the ElasticNet Model**
```python
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
```
- **`ElasticNet`**: The regression model being used.
- **`alpha=alpha`**: Controls the strength of the regularization.
- **`l1_ratio=l1_ratio`**: Determines the balance between L1 and L2 penalties.
- **`random_state=42`**: Ensures reproducibility (so the random elements behave consistently).

---

#### **3. Training the Model**
```python
lr.fit(train_x, train_y)
```
- The `fit()` method trains the model using:
  - **`train_x`**: Input features (independent variables).
  - **`train_y`**: Target values (dependent variable, `quality` in this case).
- Internally:
  - ElasticNet solves a modified linear regression cost function:
    \[
    \text{Cost Function} = \text{Sum of Squared Errors} + \alpha (\text{L1 penalty} \cdot \text{L1 ratio} + \text{L2 penalty} \cdot (1 - \text{L1 ratio}))
    \]
  - It calculates the optimal weights (coefficients) for each feature by minimizing the cost.

---

### Simplified Analogy

Think of **ElasticNet** as a team coach assigning scores to players (features):

1. The coach wants to minimize **errors** (predict the team’s total score accurately).
2. However, the coach penalizes:
   - Over-reliance on a few players (L1 regularization, sparsity).
   - Allowing some players to dominate (L2 regularization, small weights for all).
3. By balancing these penalties (via `alpha` and `l1_ratio`), the coach ensures the team (model) performs well under different conditions.

---

### How Linear Regression Fits In
ElasticNet is a **generalization of linear regression**:
- If `alpha=0`: No regularization → It’s just plain linear regression.
- By tuning `alpha` and `l1_ratio`, ElasticNet extends linear regression to make it more robust.

---
### **8. Evaluating the Model**

```python
predicted_qualities = lr.predict(test_x)
(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
```

- **Purpose**: Predicts wine quality on test data and calculates evaluation metrics.

**Example**: Like taste-testing food to ensure it meets expectations.

---

### **9. Logging Parameters and Metrics**

```python
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)
```

- **Purpose**: Logs model parameters and evaluation metrics in MLflow.
- **Why?** To track and compare experiments easily.

**Example**: Recording the ingredients and taste scores of different dishes.

---

### **10. Saving the Model**

```python
remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

if tracking_url_type_store != "file":
    mlflow.sklearn.log_model(
        lr, "model", registered_model_name="ElasticnetWineModel"
    )
else:
    mlflow.sklearn.log_model(lr, "model")
```

- **Purpose**: Saves the trained model for future use.
  - **remote_server_uri**: Sets up a remote server to log the model.
  - **log_model**: Uploads the model to MLflow's tracking server or a registry.

**Example**: Storing a recipe in a digital cookbook for future use.

---

### Summary of Workflow
1. **Data**: Wine dataset is loaded and split into training/testing sets.
2. **Model**: ElasticNet regression model is trained.
3. **Evaluation**: Metrics (RMSE, MAE, R2) are calculated.
4. **Logging**: Parameters, metrics, and the model are logged using MLflow.

**Real-World Use Case**: A data scientist compares several wine quality prediction models and uses MLflow to find and deploy the best one.

---

### What is DAGsHub?

DAGsHub is a **collaborative platform for managing machine learning projects**. Think of it as **GitHub but specifically designed for ML/AI projects**, with added tools for:
1. **Version Control for Data and Models**: Track changes in datasets, ML models, and code (using tools like DVC and Git).
2. **Experiment Tracking**: Keep a record of experiments, hyperparameters, metrics, and outputs (integrates seamlessly with MLflow).
3. **Collaboration**: Work with your team on datasets, models, and experiments just like you would with GitHub for code.

---

### Why Use DAGsHub?

ML projects are more complex than just coding:
1. **Tracking Experiments**: You tweak parameters, change models, and run experiments often. DAGsHub ensures you don't lose track of what worked and why.
2. **Versioning Data**: Unlike GitHub, DAGsHub can track large datasets and even changes made to data over time.
3. **Centralized ML Workflow**: It integrates everything—datasets, code, ML models, and experiments—in one place for easy access and collaboration.

---

### Purpose of Integrating DAGsHub with MLflow

MLflow handles **experiment tracking**, while DAGsHub provides a **platform to view, share, and manage those tracked experiments** more effectively. Here's why you might integrate them:

1. **Store and View Experiments**:
   - MLflow logs metrics, parameters, and models locally or on a remote server.
   - With DAGsHub, those logs are centralized in a **cloud-based dashboard** where your team can analyze and compare experiments.

2. **Collaboration**:
   - If you're working with a team, DAGsHub lets everyone view and contribute to the experiment logs easily, without requiring local setups.

3. **Version Control for Experiments**:
   - DAGsHub not only tracks the experiment metrics but also links them to specific code versions, dataset changes, and MLflow runs.

---

### Real-Time Example

Let’s consider a **wine quality prediction project** (like the one from the MLflow code you shared):

1. **Without DAGsHub**:
   - You run experiments on your laptop.
   - You log parameters and metrics locally with MLflow.
   - Your teammate runs a slightly different experiment on their laptop.
   - Now, both of you have separate MLflow logs, and it’s hard to combine or compare them. You also struggle to keep track of which dataset version was used.

2. **With DAGsHub**:
   - You connect MLflow to DAGsHub.
   - Every time you run an experiment, the logs (parameters, metrics, and models) are **automatically uploaded to DAGsHub**.
   - Your teammate does the same, and all experiments are now in a **shared dashboard** on DAGsHub.
   - You can both see:
     - Which dataset version was used.
     - The hyperparameters (`alpha`, `l1_ratio`) and results (`RMSE`, `R2`) for each run.
     - The code used for the experiment.
   - You can easily pick the best experiment and continue improving from there.

---

### Analogy to Understand

Imagine you're writing a group project report:

- **Without DAGsHub**: Everyone writes their section on separate Word documents. You keep emailing each other back and forth with changes. Confusing, right?
- **With DAGsHub**: Everyone writes in a shared Google Doc. All changes are visible in one place, and you can track who wrote what and when.

---

### How DAGsHub and MLflow Work Together
1. MLflow logs your experiment details locally.
2. You connect MLflow to DAGsHub using its **tracking URI**.
3. Now, whenever you run an experiment with MLflow, DAGsHub stores and displays the experiment details in an online dashboard.

---
### How to Connect MLflow to DAGsHub (Easy Step-by-Step)

Let’s break it down like a simple recipe:

---

### Ingredients:
1. **DAGsHub Account**: Sign up for free at [dagshub.com](https://dagshub.com).
2. **MLflow**: Already installed in your system.
3. **A Remote Repository**: Create a project repository on DAGsHub (just like you’d do on GitHub).

---

### Steps:

#### 1. **Create a Repository on DAGsHub**
   - Log in to DAGsHub.
   - Click **New Repository**.
   - Name it (e.g., `wine-quality-project`) and create it.
   - This repository will store your experiment logs.

---

#### 2. **Get Your Tracking URI**
   - Inside your new DAGsHub repo, go to **Settings** > **Tracking**.
   - Copy the **MLflow Tracking URI** (it will look something like `https://dagshub.com/YourUsername/YourRepoName.mlflow`).

---

#### 3. **Connect Your MLflow Code to DAGsHub**

In your MLflow script, add the following lines inside the `with mlflow.start_run()` block:

```python
# Set DAGsHub as the remote tracking server
remote_server_uri = "https://dagshub.com/YourUsername/YourRepoName.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
```

This tells MLflow to send all your logs (parameters, metrics, models) to the DAGsHub server.

---

#### 4. **Run Your MLflow Experiment**

Run your MLflow script as usual. For example:
```bash
python your_mlflow_script.py
```

After running, all your experiment logs (metrics, parameters, and models) will automatically be sent to DAGsHub.

---

#### 5. **Check Your Logs on DAGsHub**
   - Go back to your DAGsHub repo.
   - Click on **Experiments** (a tab you’ll now see in the repo).
   - You’ll see:
     - All your runs (with metrics like RMSE, MAE, etc.).
     - Hyperparameters (`alpha`, `l1_ratio`).
     - Links to models you logged.

---

### Real-Life Example

Imagine you’re running multiple experiments for a **sales prediction model**:

1. You tweak **`alpha`** and **`l1_ratio`** in your ElasticNet regression model.
2. You run 5 experiments locally and log them using MLflow.
3. DAGsHub collects all those runs in one place, showing a dashboard where:
   - You can compare metrics like RMSE or R2 for each run.
   - Your teammate can add their experiments, and you can see their results too.

---

### Why Use DAGsHub for MLflow?

- **Centralized Logs**: Instead of having experiment logs scattered across team members’ laptops, everyone’s logs are in one place.
- **Collaboration**: Everyone can view, compare, and discuss the experiments online.
- **Version Control**: It links your experiments to the exact dataset and code used, avoiding confusion.

---

![image](https://github.com/user-attachments/assets/76cb14e7-d485-4994-9614-2792d55b1cb8)

### Visual Example: Setting Up and Using DAGsHub with MLflow 🚀

Here’s a simplified walkthrough with visuals to help you connect everything smoothly.

---

### **Step 1: Create a Repository on DAGsHub**

1. **Log In** to [DAGsHub](https://dagshub.com) and click on **New Repository**.
2. **Name your repo** (e.g., `wine-quality-project`) and click **Create**.
   - Your repo will now have tabs like **Code**, **Data**, and **Experiments**.

---

### **Step 2: Get Your Tracking URI**
1. Go to your repo and click **Settings** (top-right corner).
2. Find **Tracking** in the left menu.
3. Copy the **Tracking URI**. It will look like:
   ```
   https://dagshub.com/YourUsername/YourRepoName.mlflow
   ```

---

### **Step 3: Add DAGsHub to Your MLflow Script**

In your MLflow script, add this block:
```python
# Set DAGsHub as the remote server for experiment tracking
remote_server_uri = "https://dagshub.com/YourUsername/YourRepoName.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
```

This tells MLflow to send all logs (metrics, parameters, models) to DAGsHub.

---

### **Step 4: Run Your Experiment**

Run your MLflow script as usual:
```bash
python your_mlflow_script.py
```

---

### **Step 5: View Logs on DAGsHub**
1. Go back to your DAGsHub repo.
2. Click on **Experiments**.
3. You’ll see:
   - Metrics (e.g., RMSE, R2).
   - Parameters (e.g., `alpha`, `l1_ratio`).
   - Artifacts (e.g., saved models).

---

### **What the Dashboard Looks Like**

- A table showing all runs and their metrics (e.g., RMSE for each run).
- Visualizations for comparing metrics across experiments.
- Links to view or download saved models.

---

### Real-Life Use Case: Team Collaboration

#### Scenario:
You and a teammate are building a **house price prediction model**.

1. **Your Role**:
   - You try **Linear Regression** with different parameters and log metrics (RMSE, MAE) to DAGsHub.
2. **Your Teammate's Role**:
   - They try **Random Forest** and also log their metrics to the same DAGsHub repo.
3. **On DAGsHub**:
   - Both of your results appear in the **Experiments tab**.
   - You compare which model performs better and decide the next steps—together!

---

### DAGsHub Simplified Analogy

It’s like a **Google Drive for ML experiments**:
1. MLflow is your notebook, where you jot down experiment details.
2. DAGsHub is the shared folder where everyone can access those notes and work together.

---
