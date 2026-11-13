
from pathlib import Path
import runpy
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_FILE = PROJECT_DIR / "main.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_spark_dataframe(dataframe, filename):
    """Save a Spark DataFrame as a single CSV file."""
    dataframe.toPandas().to_csv(OUTPUT_DIR / filename, index=False)


def save_bar_chart(labels, values, title, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    if not MAIN_FILE.exists():
        raise FileNotFoundError(
            "main.py was not found. Put save_outputs.py in the same folder as main.py."
        )

    if not (PROJECT_DIR / "covtype.csv").exists():
        raise FileNotFoundError(
            "covtype.csv was not found. Put covtype.csv in the same folder as main.py."
        )

    print("\nRunning main.py to generate the model results...")
    print("This will train all four models again. Please wait.\n")

    # Run the original project and collect its variables without editing main.py.
    results = runpy.run_path(str(MAIN_FILE), run_name="__main__")

    print("\nMain project completed. Saving report files...")

    # ---------------------------------------------------------
    # 1. Supervised model metrics
    # ---------------------------------------------------------
    supervised_metrics = pd.DataFrame([
        {
            "Model": "Decision Tree",
            "Accuracy": results["dt_accuracy"],
            "Weighted Precision": results["dt_precision"],
            "Weighted Recall": results["dt_recall"],
            "Weighted F1": results["dt_f1"],
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": results["lr_accuracy"],
            "Weighted Precision": results["lr_precision"],
            "Weighted Recall": results["lr_recall"],
            "Weighted F1": results["lr_f1"],
        },
    ])
    supervised_metrics.to_csv(
        OUTPUT_DIR / "supervised_model_metrics.csv", index=False
    )

    # ---------------------------------------------------------
    # 2. Confusion matrices and label mapping
    # ---------------------------------------------------------
    save_spark_dataframe(
        results["dt_confusion_matrix"],
        "decision_tree_confusion_matrix.csv",
    )
    save_spark_dataframe(
        results["lr_confusion_matrix"],
        "logistic_regression_confusion_matrix_encoded.csv",
    )

    label_mapping = pd.DataFrame([
        {"Encoded Label": index, "Actual Cover Type": cover_type}
        for index, cover_type in enumerate(
            results["label_indexer_model"].labels
        )
    ])
    label_mapping.to_csv(
        OUTPUT_DIR / "logistic_regression_label_mapping.csv", index=False
    )

    # Per-class Decision Tree metrics
    evaluator_class = results["MulticlassClassificationEvaluator"]
    dt_predictions = results["dt_predictions"]
    class_rows = []

    for cover_type in range(1, 8):
        row = {"Cover Type": cover_type}
        for metric_name, output_name in [
            ("precisionByLabel", "Precision"),
            ("recallByLabel", "Recall"),
            ("fMeasureByLabel", "F1"),
        ]:
            evaluator = evaluator_class(
                labelCol="Cover_Type",
                predictionCol="prediction",
                metricName=metric_name,
                metricLabel=float(cover_type),
            )
            row[output_name] = evaluator.evaluate(dt_predictions)
        class_rows.append(row)

    pd.DataFrame(class_rows).to_csv(
        OUTPUT_DIR / "decision_tree_class_wise_metrics.csv", index=False
    )

    # ---------------------------------------------------------
    # 3. Clustering evaluation metrics
    # ---------------------------------------------------------
    clustering_metrics = pd.DataFrame([
        {
            "Model": "K-Means",
            "Silhouette Score": results["kmeans_silhouette"],
            "Sample Size": results["kmeans_sample_count"],
            "Distance Measure": "Squared Euclidean",
        },
        {
            "Model": "Agglomerative Hierarchical",
            "Silhouette Score": results["hierarchical_silhouette"],
            "Sample Size": len(results["hierarchical_sample"]),
            "Distance Measure": "Euclidean",
        },
    ])
    clustering_metrics.to_csv(
        OUTPUT_DIR / "clustering_evaluation_metrics.csv", index=False
    )

    # ---------------------------------------------------------
    # 4. Dataset distribution and correlation data
    # ---------------------------------------------------------
    cover_data = results["cover_data"]
    cover_df = pd.DataFrame([
        {"Cover_Type": row["Cover_Type"], "Count": row["count"]}
        for row in cover_data
    ])
    cover_df.to_csv(OUTPUT_DIR / "cover_type_distribution.csv", index=False)

    numeric_columns = results["numeric_columns"]
    corr_array = results["corr_array"]
    pd.DataFrame(
        corr_array, index=numeric_columns, columns=numeric_columns
    ).to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # ---------------------------------------------------------
    # 5. K-Means cluster analysis
    # ---------------------------------------------------------
    kmeans_predictions = results["kmeans_predictions"]

    kmeans_sizes = (
        kmeans_predictions.groupBy("prediction")
        .count()
        .orderBy("prediction")
    )
    save_spark_dataframe(kmeans_sizes, "kmeans_cluster_sizes.csv")

    kmeans_environment = (
        kmeans_predictions.groupBy("prediction")
        .agg(
            results["avg"]("Elevation").alias("Average_Elevation"),
            results["avg"]("Slope").alias("Average_Slope"),
            results["avg"]("Horizontal_Distance_To_Roadways").alias(
                "Average_Distance_To_Roadways"
            ),
            results["avg"]("Horizontal_Distance_To_Hydrology").alias(
                "Average_Distance_To_Hydrology"
            ),
        )
        .orderBy("prediction")
    )
    save_spark_dataframe(
        kmeans_environment, "kmeans_environmental_summary.csv"
    )

    kmeans_cover = (
        kmeans_predictions.groupBy("prediction", "Cover_Type")
        .count()
        .orderBy("prediction", "Cover_Type")
    )
    save_spark_dataframe(
        kmeans_cover, "kmeans_cover_type_distribution.csv"
    )

    # ---------------------------------------------------------
    # 6. Hierarchical cluster analysis
    # ---------------------------------------------------------
    hierarchical_results = results["hierarchical_results"]

    hierarchical_sizes = (
        hierarchical_results.groupBy("Cluster")
        .count()
        .orderBy("Cluster")
    )
    save_spark_dataframe(
        hierarchical_sizes, "hierarchical_cluster_sizes.csv"
    )

    hierarchical_environment = (
        hierarchical_results.groupBy("Cluster")
        .agg(
            results["avg"]("Elevation").alias("Average_Elevation"),
            results["avg"]("Slope").alias("Average_Slope"),
            results["avg"]("Horizontal_Distance_To_Roadways").alias(
                "Average_Distance_To_Roadways"
            ),
            results["avg"]("Horizontal_Distance_To_Hydrology").alias(
                "Average_Distance_To_Hydrology"
            ),
        )
        .orderBy("Cluster")
    )
    save_spark_dataframe(
        hierarchical_environment,
        "hierarchical_environmental_summary.csv",
    )

    hierarchical_cover = (
        hierarchical_results.groupBy("Cluster", "Cover_Type")
        .count()
        .orderBy("Cluster", "Cover_Type")
    )
    save_spark_dataframe(
        hierarchical_cover,
        "hierarchical_cover_type_distribution.csv",
    )

    # ---------------------------------------------------------
    # 7. Save report graphs
    # ---------------------------------------------------------
    save_bar_chart(
        cover_df["Cover_Type"],
        cover_df["Count"],
        "Distribution of Forest Cover Types",
        "Cover Type",
        "Number of Records",
        "cover_type_distribution.png",
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(corr_array, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    metric_names = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
    x_positions = range(len(metric_names))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(
        [x - width / 2 for x in x_positions],
        supervised_metrics.loc[0, metric_names],
        width,
        label="Decision Tree",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        supervised_metrics.loc[1, metric_names],
        width,
        label="Logistic Regression",
    )
    plt.xticks(list(x_positions), metric_names, rotation=10)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Supervised Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "supervised_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    kmeans_sizes_pd = kmeans_sizes.toPandas()
    save_bar_chart(
        kmeans_sizes_pd["prediction"],
        kmeans_sizes_pd["count"],
        "K-Means Cluster Sizes",
        "Cluster",
        "Number of Records",
        "kmeans_cluster_sizes.png",
    )

    hierarchical_sizes_pd = hierarchical_sizes.toPandas()
    save_bar_chart(
        hierarchical_sizes_pd["Cluster"],
        hierarchical_sizes_pd["count"],
        "Agglomerative Hierarchical Cluster Sizes",
        "Cluster",
        "Number of Records",
        "hierarchical_cluster_sizes.png",
    )

    print("\nAll report outputs were saved successfully!")
    print("Output folder:", OUTPUT_DIR)
    print("You can now open the outputs folder and use the CSV files and PNG graphs in your report.")

    # Stop Spark only after all output files have been created.
    results["spark"].stop()


if __name__ == "__main__":
    main()
