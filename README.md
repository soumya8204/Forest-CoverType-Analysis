
# Forest CoverType Classification and Environmental Pattern Discovery Using PySpark

## Project Overview

This project applies distributed data processing and machine learning techniques to the Forest CoverType dataset.

It uses Apache Spark (PySpark) for data preparation, exploratory data analysis, supervised classification, and K-Means clustering. Scikit-learn is used for Agglomerative Hierarchical Clustering on a representative sample.

The project has two main goals:

1. Predict forest cover types using environmental and cartographic features.
2. Discover patterns and groups of observations with similar environmental characteristics.

The dataset contains 581,012 observations, 54 input features, and 7 forest cover classes.

## Objectives

- Perform exploratory data analysis on the dataset.
- Identify missing values and duplicate records.
- Analyze forest cover type distribution.
- Study correlations among environmental features.
- Apply two supervised learning algorithms.
- Apply two unsupervised learning algorithms.
- Evaluate classification and clustering performance.
- Analyze environmental patterns across clusters.

## Dataset

**Dataset:** Covertype (Forest Cover Type)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/31/covertype)

The dataset contains cartographic information describing forest areas in northern Colorado.

Features include:

- Elevation, Aspect, and Slope
- Distance to Hydrology
- Distance to Roadways
- Hillshade measurements
- Distance to Fire Points
- Wilderness Area indicators
- Soil Type indicators

The target variable is `Cover_Type`, which represents seven forest cover classes.

| Property | Value |
|---|---:|
| Records | 581,012 |
| Total columns | 55 |
| Input features | 54 |
| Target classes | 7 |

The dataset is originally distributed without column headers. Descriptive column names are assigned during preprocessing.

## Machine Learning Algorithms

### Supervised Learning

**1. Decision Tree Classifier**

- Implemented using PySpark MLlib.
- Uses decision rules to classify forest cover types.
- Maximum depth: 10.

**2. Logistic Regression**

- Implemented using PySpark MLlib.
- Performs multiclass classification.
- Includes label indexing and feature scaling.
- Maximum iterations: 50.
- Regularization parameter: 0.01.

### Unsupervised Learning

**3. K-Means Clustering**

- Implemented using PySpark MLlib.
- Groups observations based on feature similarity.
- Number of clusters: 7.
- Maximum iterations: 20.

**4. Agglomerative Hierarchical Clustering**

- Implemented using scikit-learn.
- Uses Ward linkage.
- Applied to a representative sample of up to 2,000 observations due to computational requirements.

Note: Cluster identifiers are arbitrary and do not directly correspond to the original forest cover type labels.

## Methodology

The project follows these steps:

1. Load the Covertype dataset using PySpark.
2. Assign descriptive column names.
3. Perform exploratory data analysis.
4. Check missing values and duplicates.
5. Analyze class distribution and feature correlations.
6. Assemble the input features into a feature vector.
7. Split the dataset into 80% training and 20% testing data.
8. Train Decision Tree and Logistic Regression models.
9. Apply K-Means and Agglomerative Hierarchical Clustering.
10. Evaluate model performance and analyze cluster characteristics.
11. Generate graphs and output files for reporting.

The target column is excluded from clustering input features and used only for interpreting the resulting clusters.

## Evaluation Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Clustering

- Silhouette Score
- Cluster Size Distribution
- Environmental Feature Summaries
- Cover Type Distribution Across Clusters

Because the dataset is imbalanced, classification accuracy is considered alongside weighted and class-wise metrics.

## Tools and Technologies

- Python 3.11
- Apache Spark 3.5.9
- PySpark MLlib
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Visual Studio Code

## Project Structure

```text
Forest-CoverType-Analysis/
│
├── main.py
├── outputs.py
├── covtype.csv
├── README.md
│
└── outputs/
    ├── supervised_model_metrics.csv
    ├── decision_tree_confusion_matrix.csv
    ├── logistic_regression_confusion_matrix_encoded.csv
    ├── clustering_evaluation_metrics.csv
    ├── cover_type_distribution.csv
    ├── correlation_matrix.csv
    ├── kmeans_cluster_sizes.csv
    ├── hierarchical_cluster_sizes.csv
    ├── cover_type_distribution.png
    ├── correlation_heatmap.png
    ├── supervised_model_comparison.png
    ├── kmeans_cluster_sizes.png
    └── hierarchical_cluster_sizes.png
```

## Setup and Execution

### Install dependencies

Run in the VS Code CMD terminal:

```cmd
"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" -m pip install pyspark==3.5.9 scikit-learn numpy pandas matplotlib
```

### Configure Spark

```cmd
set SPARK_HOME=C:\spark
set PATH=%SPARK_HOME%\bin;%PATH%
set PYSPARK_PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe
set PYSPARK_DRIVER_PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe
```

### Place the dataset

Download the Covertype dataset from the UCI repository and place the CSV file in the same directory as `main.py`.

Name the file:

`covtype.csv`

### Run the project

```cmd
"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" main.py
```

### Generate report outputs

```cmd
"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" save_outputs.py
```

The output-saving script generates evaluation tables and visualizations inside the `outputs` directory.

## Results

Selected results from the completed project run:

| Model | Evaluation | Result |
|---|---|---:|
| Decision Tree | Accuracy | 77.97% |
| K-Means | Silhouette Score | 0.1520 |
| Agglomerative Hierarchical Clustering | Silhouette Score | 0.2313 |

The Decision Tree achieved approximately 77.97% accuracy on the test dataset.

The clustering scores represent cluster separation under their respective evaluation procedures. They are not classification accuracies and should be compared carefully because the implementations use different evaluation settings and sample sizes.

Refer to the generated CSV files and graphs in the `outputs` folder for detailed results.

## Limitations

- The dataset contains imbalanced target classes.
- Minority cover types are more difficult to classify.
- Cluster labels do not directly represent forest cover types.
- Hierarchical clustering is performed on a limited sample due to computational requirements.
- Clustering results depend on sampling, scaling, and model parameters.
- Agglomerative Hierarchical Clustering does not provide a standard prediction method for unseen observations.

## Conclusion

This project demonstrates how PySpark and machine learning can be used to analyze a large environmental dataset.

The supervised models classify forest cover types using environmental features, while unsupervised clustering techniques help discover groups of observations with similar characteristics.

Combining classification, clustering, exploratory analysis, and visualization provides different perspectives on the Forest CoverType dataset.

## References

1. UCI Machine Learning Repository – Covertype Dataset: https://archive.ics.uci.edu/dataset/31/covertype
2. Apache Spark – MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
3. Apache Spark – Classification and Regression: https://spark.apache.org/docs/latest/ml-classification-regression.html
4. Scikit-learn – Agglomerative Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
5. Scikit-learn – Silhouette Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
