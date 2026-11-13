# ============================================================
# FOREST COVERTYPE CLASSIFICATION AND ENVIRONMENTAL PATTERN
# DISCOVERY USING PYSPARK
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# === PySpark SQL Libraries ===

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    sum as spark_sum,
    avg
)


# === Machine Learning Feature Libraries ===

from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    StandardScaler
)


# === Statistical Analysis Libraries ===

from pyspark.ml.stat import Correlation


# === Supervised Learning Algorithms ===

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression
)


# === Unsupervised Learning Algorithms ===

from pyspark.ml.clustering import KMeans


# === Model Evaluation Libraries ===

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    ClusteringEvaluator
)


# === Hierarchical Clustering Libraries ===

import numpy as np

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score


# === Visualisation Libraries ===

import matplotlib.pyplot as plt


# ============================================================
# 2. CREATE SPARK SESSION
# ============================================================

# === Initialise Spark Session ===

spark = SparkSession.builder \
    .appName("Forest CoverType Classification") \
    .getOrCreate()

print("Spark Session Created Successfully!")


# ============================================================
# 3. LOAD DATASET
# ============================================================

# === Read CSV Dataset ===

data = spark.read.csv(
    "covtype.csv",
    header=False,
    inferSchema=True
)

print("Dataset Loaded Successfully!")


# ============================================================
# 4. DATASET UNDERSTANDING
# ============================================================

# === Check Dataset Dimensions ===

print("\nDataset Dimensions:")

print("Total Rows:", data.count())

print("Total Columns:", len(data.columns))


# === Assign Column Names ===

columns = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]


# --- Add Wilderness Area Columns ---

columns += [
    f"Wilderness_Area_{i}"
    for i in range(1, 5)
]


# --- Add Soil Type Columns ---

columns += [
    f"Soil_Type_{i}"
    for i in range(1, 41)
]


# --- Add Target Column ---

columns += ["Cover_Type"]

data = data.toDF(*columns)

print("\nColumn Names Assigned Successfully!")


# === Display Dataset ===

print("\nFirst 5 Records:")

data.show(5)


# === Check Data Types ===

print("\nDataset Schema:")

data.printSchema()


# === Check Missing Values ===

print("\nMissing Values:")

missing_values = data.select([

    spark_sum(
        col(c).isNull().cast("int")
    ).alias(c)

    for c in data.columns

])

missing_values.show(truncate=False)


# === Check Target Variable Distribution ===

print("\nCover Type Distribution:")

data.groupBy("Cover_Type") \
    .count() \
    .orderBy("Cover_Type") \
    .show()


# === Check Summary Statistics ===

print("\nSummary Statistics:")

data.describe().show()


# === Check Duplicate Records ===

print("\nChecking Duplicate Records:")

total_records = data.count()

unique_records = data.dropDuplicates().count()

duplicate_records = total_records - unique_records

print("Total Duplicate Records:", duplicate_records)


# === Display Final Dataset Information ===

print("\nFinal Dataset Information")

print("Rows:", total_records)

print("Columns:", len(data.columns))

print("Feature Columns:", len(data.columns) - 1)

print("Target Column: Cover_Type")


# ============================================================
# 5. DATA VISUALISATION
# ============================================================

# === Visualise Cover Type Distribution ===

cover_data = data.groupBy("Cover_Type") \
    .count() \
    .orderBy("Cover_Type") \
    .collect()

cover_types = [
    row["Cover_Type"]
    for row in cover_data
]

counts = [
    row["count"]
    for row in cover_data
]

plt.bar(cover_types, counts)

plt.xlabel("Cover Type")

plt.ylabel("Number of Records")

plt.title("Distribution of Forest Cover Types")

plt.show()


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

# === Select Numerical Features ===

numeric_columns = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]


# === Assemble Numerical Features ===

correlation_assembler = VectorAssembler(
    inputCols=numeric_columns,
    outputCol="correlation_features"
)

feature_data = correlation_assembler.transform(data)


# === Calculate Pearson Correlation ===

correlation_matrix = Correlation.corr(
    feature_data,
    "correlation_features",
    "pearson"
).head()[0]

print("\nCorrelation Matrix:")

print(correlation_matrix)


# === Generate Correlation Heatmap ===

corr_array = correlation_matrix.toArray()

plt.figure(figsize=(12, 8))

plt.imshow(
    corr_array,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.show()


# ============================================================
# 7. FEATURE ANALYSIS BY COVER TYPE
# ============================================================

# === Average Elevation ===

print("\nAverage Elevation by Cover Type:")

data.groupBy("Cover_Type").agg(
    {"Elevation": "mean"}
).show()


# === Average Slope ===

print("\nAverage Slope by Cover Type:")

data.groupBy("Cover_Type").agg(
    {"Slope": "mean"}
).show()


# === Average Distance to Roadways ===

print("\nAverage Distance to Roadways by Cover Type:")

data.groupBy("Cover_Type").agg(
    {"Horizontal_Distance_To_Roadways": "mean"}
).show()


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

# === Split Dataset into Training and Testing Data ===

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining Records:", train_data.count())

print("Testing Records:", test_data.count())


# === Select Feature Columns ===

feature_columns = [
    c for c in data.columns
    if c != "Cover_Type"
]

print("\nNumber of Features:", len(feature_columns))

print("Target Column: Cover_Type")


# === Feature Vector Assembly ===

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

train_data = assembler.transform(train_data)

test_data = assembler.transform(test_data)

print("\nFeature Vector Created Successfully!")


# === Display Feature Vector ===

train_data.select(
    "features",
    "Cover_Type"
).show(5, truncate=False)


# === Check Preprocessed Dataset Schema ===

train_data.printSchema()


# ============================================================
# 9. SUPERVISED LEARNING
# ============================================================


# ============================================================
# 9.1 DECISION TREE CLASSIFIER
# ============================================================

# === Create Decision Tree Model ===

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="Cover_Type",
    maxDepth=10,
    seed=42
)


# === Train Decision Tree Model ===

dt_model = dt.fit(train_data)

print("\nDecision Tree Model Trained Successfully!")


# ============================================================
# 9.2 DECISION TREE PREDICTIONS
# ============================================================

# === Generate Predictions ===

dt_predictions = dt_model.transform(test_data)


# === Display Predictions ===

print("\nDecision Tree Predictions:")

dt_predictions.select(
    "Cover_Type",
    "prediction",
    "probability"
).show(10, truncate=False)


# ============================================================
# 9.3 DECISION TREE MODEL EVALUATION
# ============================================================

# === Calculate Accuracy ===

dt_accuracy = MulticlassClassificationEvaluator(
    labelCol="Cover_Type",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(dt_predictions)


# === Calculate Weighted Precision ===

dt_precision = MulticlassClassificationEvaluator(
    labelCol="Cover_Type",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(dt_predictions)


# === Calculate Weighted Recall ===

dt_recall = MulticlassClassificationEvaluator(
    labelCol="Cover_Type",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(dt_predictions)


# === Calculate Weighted F1 Score ===

dt_f1 = MulticlassClassificationEvaluator(
    labelCol="Cover_Type",
    predictionCol="prediction",
    metricName="f1"
).evaluate(dt_predictions)


# === Display Decision Tree Results ===

print("\nDECISION TREE RESULTS")

print("Accuracy:", dt_accuracy)

print("Accuracy (%):", dt_accuracy * 100)

print("Weighted Precision:", dt_precision)

print("Weighted Recall:", dt_recall)

print("Weighted F1 Score:", dt_f1)


# ============================================================
# 9.4 DECISION TREE CONFUSION MATRIX
# ============================================================

# === Generate Confusion Matrix ===

print("\nDecision Tree Confusion Matrix:")

dt_predictions.groupBy(
    "Cover_Type",
    "prediction"
).count().orderBy(
    "Cover_Type",
    "prediction"
).show(100)


# === Display Pivoted Confusion Matrix ===

dt_confusion_matrix = dt_predictions.groupBy(
    "Cover_Type"
).pivot(
    "prediction",
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
).count().na.fill(0).orderBy("Cover_Type")

dt_confusion_matrix.show()


# ============================================================
# 9.5 DECISION TREE CLASS-WISE EVALUATION
# ============================================================

# === Calculate Per-Class Precision, Recall and F1 ===

print("\nDecision Tree Per-Class Metrics:")

for label in range(1, 8):

    class_precision = MulticlassClassificationEvaluator(
        labelCol="Cover_Type",
        predictionCol="prediction",
        metricName="precisionByLabel",
        metricLabel=float(label)
    ).evaluate(dt_predictions)

    class_recall = MulticlassClassificationEvaluator(
        labelCol="Cover_Type",
        predictionCol="prediction",
        metricName="recallByLabel",
        metricLabel=float(label)
    ).evaluate(dt_predictions)

    class_f1 = MulticlassClassificationEvaluator(
        labelCol="Cover_Type",
        predictionCol="prediction",
        metricName="fMeasureByLabel",
        metricLabel=float(label)
    ).evaluate(dt_predictions)

    print(
        "Class:", label,
        "| Precision:", round(class_precision, 4),
        "| Recall:", round(class_recall, 4),
        "| F1:", round(class_f1, 4)
    )


# ============================================================
# 9.6 LOGISTIC REGRESSION
# ============================================================

# === Convert Target Labels into Numerical Indices ===

label_indexer = StringIndexer(
    inputCol="Cover_Type",
    outputCol="label"
)

label_indexer_model = label_indexer.fit(train_data)

train_indexed = label_indexer_model.transform(train_data)

test_indexed = label_indexer_model.transform(test_data)

print("\nLabel Encoding Completed!")

train_indexed.select(
    "Cover_Type",
    "label"
).show(10)


# === Feature Scaling ===

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(train_indexed)

train_scaled = scaler_model.transform(train_indexed)

test_scaled = scaler_model.transform(test_indexed)

print("\nFeature Scaling Completed!")


# === Create Logistic Regression Model ===

lr = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=50,
    regParam=0.01
)


# === Train Logistic Regression Model ===

lr_model = lr.fit(train_scaled)

print("\nLogistic Regression Model Trained Successfully!")


# ============================================================
# 9.7 LOGISTIC REGRESSION PREDICTIONS
# ============================================================

# === Generate Predictions ===

lr_predictions = lr_model.transform(test_scaled)


# === Display Predictions ===

print("\nLogistic Regression Predictions:")

lr_predictions.select(
    "Cover_Type",
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)


# ============================================================
# 9.8 LOGISTIC REGRESSION MODEL EVALUATION
# ============================================================

# === Calculate Accuracy ===

lr_accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(lr_predictions)


# === Calculate Weighted Precision ===

lr_precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(lr_predictions)


# === Calculate Weighted Recall ===

lr_recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(lr_predictions)


# === Calculate Weighted F1 Score ===

lr_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(lr_predictions)


# === Display Logistic Regression Results ===

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy:", lr_accuracy)

print("Accuracy (%):", lr_accuracy * 100)

print("Weighted Precision:", lr_precision)

print("Weighted Recall:", lr_recall)

print("Weighted F1 Score:", lr_f1)


# ============================================================
# 9.9 LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

# === Generate Confusion Matrix ===

lr_confusion_matrix = lr_predictions.groupBy(
    "label"
).pivot(
    "prediction",
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
).count().na.fill(0).orderBy("label")

lr_confusion_matrix.show()


# === Display Label Mapping ===

print("\nLogistic Regression Label Mapping:")

for index, cover_type in enumerate(label_indexer_model.labels):
    print(
        "Encoded Label:", index,
        "= Cover Type:", cover_type
    )


# ============================================================
# 9.10 COMPARE SUPERVISED LEARNING MODELS
# ============================================================

# === Compare Decision Tree and Logistic Regression ===

print("\n========== SUPERVISED MODEL COMPARISON ==========")

print(
    "{:<25} {:<20} {:<20}".format(
        "Metric",
        "Decision Tree",
        "Logistic Regression"
    )
)

print("-" * 65)

print(
    "{:<25} {:<20.4f} {:<20.4f}".format(
        "Accuracy",
        dt_accuracy,
        lr_accuracy
    )
)

print(
    "{:<25} {:<20.4f} {:<20.4f}".format(
        "Weighted Precision",
        dt_precision,
        lr_precision
    )
)

print(
    "{:<25} {:<20.4f} {:<20.4f}".format(
        "Weighted Recall",
        dt_recall,
        lr_recall
    )
)

print(
    "{:<25} {:<20.4f} {:<20.4f}".format(
        "Weighted F1",
        dt_f1,
        lr_f1
    )
)


# ============================================================
# 10. UNSUPERVISED LEARNING
# ============================================================


# ============================================================
# 10.1 K-MEANS CLUSTERING
# ============================================================

print("\n========== K-MEANS CLUSTERING ==========")


# === Prepare Data for Clustering ===

# Cover_Type is excluded from the clustering features.

kmeans_data = train_scaled.select(
    "scaledFeatures",
    "Cover_Type",
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Hydrology"
)


print("\nData Prepared for K-Means Clustering!")


# === Create Sample for Clustering ===

# Use 20% of training data to reduce computation time.

kmeans_sample = kmeans_data.sample(
    withReplacement=False,
    fraction=0.20,
    seed=42
).cache()


kmeans_sample_count = kmeans_sample.count()

print(
    "\nNumber of Records Used for Clustering:",
    kmeans_sample_count
)


# ============================================================
# 10.2 CREATE K-MEANS MODEL
# ============================================================

# === Initialise K-Means Algorithm ===

kmeans = KMeans(
    featuresCol="scaledFeatures",
    predictionCol="prediction",
    k=7,
    maxIter=20,
    seed=42
)


# ============================================================
# 10.3 TRAIN K-MEANS MODEL
# ============================================================

# === Fit K-Means Model ===

kmeans_model = kmeans.fit(kmeans_sample)

print("\nK-Means Model Trained Successfully!")


# === Display Cluster Centres ===

print("\nCluster Centres:")

for i, center in enumerate(kmeans_model.clusterCenters()):
    print("Cluster", i, ":", center)


# === Display Within Set Sum of Squared Errors ===

print("\nWithin Set Sum of Squared Errors:")

print(kmeans_model.summary.trainingCost)


# ============================================================
# 10.4 K-MEANS CLUSTER PREDICTIONS
# ============================================================

# === Assign Clusters to Records ===

kmeans_predictions = kmeans_model.transform(kmeans_sample)


# === Display Cluster Predictions ===

print("\nK-Means Cluster Predictions:")

kmeans_predictions.select(
    "Cover_Type",
    "prediction"
).show(10)


# ============================================================
# 10.5 K-MEANS MODEL EVALUATION
# ============================================================

# === Calculate Silhouette Score ===

clustering_evaluator = ClusteringEvaluator(
    featuresCol="scaledFeatures",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

kmeans_silhouette = clustering_evaluator.evaluate(
    kmeans_predictions
)


# === Display Evaluation Results ===

print("\nK-MEANS CLUSTERING RESULTS")

print("Silhouette Score:", kmeans_silhouette)


# ============================================================
# 10.6 CLUSTER DISTRIBUTION
# ============================================================

# === Count Records in Each Cluster ===

print("\nNumber of Records in Each Cluster:")

kmeans_predictions.groupBy(
    "prediction"
).count().orderBy(
    "prediction"
).show()


# ============================================================
# 10.7 ENVIRONMENTAL PATTERN ANALYSIS
# ============================================================

# === Calculate Average Environmental Features
#     Within Each Cluster ===

print("\nEnvironmental Characteristics of Each Cluster:")

kmeans_predictions.groupBy(
    "prediction"
).agg(

    avg("Elevation").alias("Average_Elevation"),

    avg("Slope").alias("Average_Slope"),

    avg("Horizontal_Distance_To_Roadways").alias(
        "Average_Distance_To_Roadways"
    ),

    avg("Horizontal_Distance_To_Hydrology").alias(
        "Average_Distance_To_Hydrology"
    )

).orderBy(
    "prediction"
).show()


# ============================================================
# 10.8 COMPARE CLUSTERS WITH ACTUAL COVER TYPES
# ============================================================

# === Examine Cover Type Distribution Within Each Cluster ===

# Cover_Type is used ONLY for interpretation.
# It was NOT used to train the K-Means model.

print("\nCover Type Distribution Within Each Cluster:")

kmeans_predictions.groupBy(
    "prediction",
    "Cover_Type"
).count().orderBy(
    "prediction",
    "Cover_Type"
).show(100)


# === Release Cached Sample ===

kmeans_sample.unpersist()


# ============================================================
# 11. AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ============================================================

print("\n========== AGGLOMERATIVE HIERARCHICAL CLUSTERING ==========")


# ============================================================
# 11.1 PREPARE DATA FOR HIERARCHICAL CLUSTERING
# ============================================================

# === Select Environmental Features ===

# Cover_Type is retained only for later interpretation.
# It is NOT included in the clustering feature vectors.

hierarchical_data = train_scaled.select(
    "scaledFeatures",
    "Cover_Type",
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Hydrology"
)


# === Create a Small Sample ===

# Hierarchical clustering is computationally expensive.
# Use a maximum of 2,000 records.

hierarchical_sample = hierarchical_data.sample(
    withReplacement=False,
    fraction=0.02,
    seed=42
).limit(2000).collect()


print(
    "\nNumber of Records Used:",
    len(hierarchical_sample)
)


# ============================================================
# 11.2 CONVERT DATA INTO NUMPY ARRAY
# ============================================================

# === Extract Feature Vectors ===

X_hierarchical = np.array([

    row["scaledFeatures"].toArray()

    for row in hierarchical_sample

])


print("\nFeature Matrix Shape:", X_hierarchical.shape)


# ============================================================
# 11.3 CREATE HIERARCHICAL CLUSTERING MODEL
# ============================================================

# === Initialise Agglomerative Clustering ===

hierarchical_model = AgglomerativeClustering(
    n_clusters=7,
    linkage="ward"
)


# ============================================================
# 11.4 TRAIN HIERARCHICAL CLUSTERING MODEL
# ============================================================

# === Fit Model and Assign Clusters ===

hierarchical_labels = hierarchical_model.fit_predict(
    X_hierarchical
)


print("\nAgglomerative Hierarchical Clustering Completed!")


# ============================================================
# 11.5 HIERARCHICAL CLUSTER PREDICTIONS
# ============================================================

# === Combine Cluster Labels with Original Information ===

hierarchical_results = spark.createDataFrame(

    [

        (
            int(hierarchical_labels[i]),
            int(row["Cover_Type"]),
            float(row["Elevation"]),
            float(row["Slope"]),
            float(row["Horizontal_Distance_To_Roadways"]),
            float(row["Horizontal_Distance_To_Hydrology"])

        )

        for i, row in enumerate(hierarchical_sample)

    ],

    [
        "Cluster",
        "Cover_Type",
        "Elevation",
        "Slope",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Hydrology"
    ]

)


# === Display Cluster Predictions ===

print("\nHierarchical Clustering Predictions:")

hierarchical_results.show(10)


# ============================================================
# 11.6 HIERARCHICAL CLUSTERING MODEL EVALUATION
# ============================================================

# === Calculate Silhouette Score ===

hierarchical_silhouette = silhouette_score(
    X_hierarchical,
    hierarchical_labels,
    metric="euclidean"
)


# === Display Evaluation Results ===

print("\nHIERARCHICAL CLUSTERING RESULTS")

print("Silhouette Score:", hierarchical_silhouette)


# ============================================================
# 11.7 CLUSTER DISTRIBUTION
# ============================================================

# === Count Records in Each Cluster ===

print("\nNumber of Records in Each Cluster:")

hierarchical_results.groupBy(
    "Cluster"
).count().orderBy(
    "Cluster"
).show()


# ============================================================
# 11.8 ENVIRONMENTAL PATTERN ANALYSIS
# ============================================================

# === Calculate Average Environmental Features
#     Within Each Cluster ===

print("\nEnvironmental Characteristics of Each Cluster:")

hierarchical_results.groupBy(
    "Cluster"
).agg(

    avg("Elevation").alias("Average_Elevation"),

    avg("Slope").alias("Average_Slope"),

    avg("Horizontal_Distance_To_Roadways").alias(
        "Average_Distance_To_Roadways"
    ),

    avg("Horizontal_Distance_To_Hydrology").alias(
        "Average_Distance_To_Hydrology"
    )

).orderBy(
    "Cluster"
).show()


# ============================================================
# 11.9 COMPARE CLUSTERS WITH ACTUAL COVER TYPES
# ============================================================

# === Examine Cover Type Distribution Within Each Cluster ===

# Cover_Type is used ONLY for interpretation.
# It was NOT used to train the clustering model.

print("\nCover Type Distribution Within Each Cluster:")

hierarchical_results.groupBy(
    "Cluster",
    "Cover_Type"
).count().orderBy(
    "Cluster",
    "Cover_Type"
).show(100)


# ============================================================
# 12. MODEL COMPARISON AND FINAL ANALYSIS
# ============================================================

# === Display All Four Models ===

print("\n========== FINAL PROJECT MODEL SUMMARY ==========")

print("1. Decision Tree Classification")

print("2. Logistic Regression")

print("3. K-Means Clustering")

print("4. Agglomerative Hierarchical Clustering")


# === Display Supervised Model Comparison ===

print("\nSUPERVISED MODEL RESULTS")

print("Decision Tree Accuracy:", dt_accuracy)

print("Logistic Regression Accuracy:", lr_accuracy)


# === Display Unsupervised Model Comparison ===

print("\nUNSUPERVISED MODEL RESULTS")

print("K-Means Silhouette Score:", kmeans_silhouette)

print(
    "Hierarchical Clustering Silhouette Score:",
    hierarchical_silhouette
)



