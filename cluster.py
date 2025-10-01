# Import necessary functions
import pandas as pd 
import numpy as np
from datetime import datetime
# from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import (col as F_col, approx_percentile, mean, rank, percentile_cont, max as F_max, 
                                          year, avg, count_distinct, date_trunc, ltrim, coalesce, sql_expr, 
                                          min as F_min, to_date, current_date, sum as F_sum, dateadd, lag, 
                                          lower, datediff, when, lit, row_number, coalesce, count, 
                                          round as sf_round, exp)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed

class build_cluster():
    def __init__(self, clusters, data, session, runid, runtimestamp, min_customers=60, features=["RECENCY_DAYS", "ORDER_FREQUENCY", "CUSTOMER_AGE_YEARS", "AVG_ORDER_VALUE"]):
        self.min_customers = min_customers
        self.clusters = clusters
        self.data = data
        self.session = session
        self.runid = runid
        self.runtimestamp= runtimestamp
        self.features = features

    def filter_df(self):
        # print("Starting...")
        self.cluster_df = self.data.select("CUSTOMER_ID", "SEGMENT", "RECENCY_DAYS",
                  "ORDER_FREQUENCY", "CUSTOMER_AGE_YEARS",
                  "AVG_ORDER_VALUE", "CONTRACT_STATUS").distinct()
        # print("Spark df len:", self.cluster_df.count())
        self.pdcluster_df = self.cluster_df.to_pandas()
        # self.pdcluster_df = self.cluster_df.limit(1000).to_pandas() ################ to remove limit
        self.cluster_df = None
        self.pdcluster_df = self.pdcluster_df.dropna(subset=["SEGMENT", "RECENCY_DAYS", "ORDER_FREQUENCY", "AVG_ORDER_VALUE","CUSTOMER_AGE_YEARS","CONTRACT_STATUS"])
        # print("Limited data to: ", self.pdcluster_df.count())
    
    def compute_cv_summary(self):
        grouped = self.pdcluster_df.groupby("SEGMENT").agg(
        recency_cv=("RECENCY_DAYS", lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan),
        frequency_cv=("ORDER_FREQUENCY", lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan),
        aov_cv=("AVG_ORDER_VALUE", lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan),
        customer_age_cv=("CUSTOMER_AGE_YEARS", lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan),
        customer_count=("CUSTOMER_ID", "count")
        ).reset_index()
        grouped["avg_cv"] = grouped[["recency_cv", "frequency_cv", "aov_cv", "customer_age_cv"]].mean(axis=1)
        grouped["should_cluster"] = (grouped["avg_cv"] > 1) & (grouped["customer_count"] >= self.min_customers)

        # Merge CV summary stats back onto behavioral_df using SEGMENT
        grouped_enriched = self.pdcluster_df.merge(
            grouped[[
                "SEGMENT",
                "recency_cv",
                "frequency_cv",
                "aov_cv",
                "customer_age_cv",
                "customer_count",
                "avg_cv",
                "should_cluster"
            ]],
            on="SEGMENT",
            how="left"
        )
        self.grouped_enriched = grouped_enriched.reset_index()
        # print("Created cv summary")
        self.pdcluster_df = None
        
    def create_cluster_df(self):
        df_scaled_parts = []
        for segment in self.grouped_enriched["SEGMENT"].unique():
            segment_df = self.grouped_enriched[self.grouped_enriched["SEGMENT"] == segment].drop_duplicates().copy()
            segment_df["CONTRACT_STATUS_FLAG"] = (segment_df["CONTRACT_STATUS"] == "Y").astype(int)

            # Apply log1p transformation to skewed features (log(x + 1))
            # The log1p transformation is applied only if the feature exists in the 'features' list.
            if "RECENCY_DAYS" in self.features:
                segment_df["RECENCY_DAYS"] = np.log1p(segment_df["RECENCY_DAYS"])
            if "ORDER_FREQUENCY" in self.features:
                segment_df["ORDER_FREQUENCY"] = np.log1p(segment_df["ORDER_FREQUENCY"])
            if "AVG_ORDER_VALUE" in self.features:
                # Apply log1p transformation to 'AVG_ORDER_VALUE', ensuring no log(0) or log(negative)
                segment_df["AVG_ORDER_VALUE"] = np.log1p(segment_df["AVG_ORDER_VALUE"].clip(lower=1))
            
            # Select the features to be scaled, which includes the scaled CONTRACT_STATUS_FLAG
            segment_features = segment_df[self.features + ["CONTRACT_STATUS_FLAG"]].copy()

            # Initialize the RobustScaler, which is less sensitive to outliers
            scaler = RobustScaler()

            # Scale the features: The scaler returns a 2D numpy array of scaled values
            scaled_array = scaler.fit_transform(segment_features)

            # Assign scaled values back to the segment_df DataFrame with new column names
            for i, feat in enumerate(self.features + ["CONTRACT_STATUS_FLAG"]):
                segment_df[f"{feat}_SCALED"] = scaled_array[:, i]  # Assign each scaled feature

            # Append the processed segment DataFrame to the list
            df_scaled_parts.append(segment_df)

        # Concatenate all the segment DataFrames back into one DataFrame
        # ignore_index=True resets the index after concatenation to avoid duplicate indices
        self.df_scaled_parts = pd.concat(df_scaled_parts, ignore_index=True)
        # print("Created cluster df")
        self.grouped_enriched = None
        
    def get_scaled_feature_names(self):
        self.scaled_feature_names = [f"{f}_SCALED" for f in self.features] + ["CONTRACT_STATUS_FLAG_SCALED"]
        # print("Scaled feature names")
        
    def find_optimal_k_using_silhouette(self):
        """
        Function to find the optimal number of clusters (K) using silhouette scores for each segment.
        Only returns the silhouette-based best k value and writes the results to a Snowflake table.

        Args:
        pdf_scaled (pd.DataFrame): The preprocessed Pandas DataFrame with scaled features.
        scaled_feature_names (list): The list of feature names that have been scaled.
        session (snowflake.snowpark.Session): The Snowflake session object.

        Returns:
        pd.DataFrame: The DataFrame with the silhouette-based best K for each segment.
        """
        # print("Finding optimal k ...")
        cluster_results = []  # List to store the clustering results for each segment

        segments_to_cluster = self.df_scaled_parts[self.df_scaled_parts["should_cluster"] == True]["SEGMENT"].unique()

        total_segments = len(segments_to_cluster)
        print(f"Total segments to cluster: {total_segments}")

        # Loop over segments to perform clustering
        for idx, segment in enumerate(segments_to_cluster, 1):
            # print("Processing segment:", segment)
            segment_df = self.df_scaled_parts[self.df_scaled_parts["SEGMENT"] == segment]
            X = segment_df[self.scaled_feature_names].dropna()  # Remove rows with missing values

            if idx == 1 or idx % 100 == 0:
                print(f"[{idx}/{total_segments}] Finding optimal k segment: {segment} with {len(X)} customers")

            k_values = list(range(self.clusters[0], min(self.clusters[1]+1, len(X))))  # Possible values for K (2 to 10)
            silhouette_scores = []  # List to store silhouette scores
            best_k_silhouette = None
            best_silhouette_score = -1

            # Run KMeans for each value of k
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                # Calculate silhouette score for k > 1
                if k > 1:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                    if score > best_silhouette_score:
                        best_silhouette_score = score
                        best_k_silhouette = k

            # Append results for the current segment
            cluster_results.append({
                "SEGMENT": segment,
                "customer_count": len(X),
                "best_k_silhouette": best_k_silhouette,
                "best_silhouette_score": best_silhouette_score
            })

        # Create a DataFrame from the clustering results
        df_cluster_k_results = pd.DataFrame(cluster_results)

        df_cluster_k_results["RUNID"] = self.runid
        df_cluster_k_results["RUNTIMESTAMP"] = self.runtimestamp
        # Convert the results DataFrame to a Snowpark DataFrame
        self.best_k = df_cluster_k_results
        self.snowpark_df = self.session.create_dataframe(df_cluster_k_results)

    def save_optimal_k(self, table):
        # Write the results into the Snowflake table
        self.snowpark_df.write.mode("append").save_as_table(table)
        # print(f"Results saved to Snowflake table: {table}")
        
    def temp_optimal_k(self, table):
        self.snowpark_df.createOrReplaceTempView(table)

    def assign_clusters_to_customers(self):
        """
        Assigns the optimal cluster label to each customer in the DataFrame based on the best K value
        for their respective segment, using silhouette score to determine the best K.
        Writes the resulting DataFrame with clusters to a Snowflake table.

        Args:
        pdf_scaled (DataFrame): The Pandas DataFrame with scaled features.
        best_k (DataFrame): DataFrame containing the optimal K for each segment based on silhouette score.
        scaled_feature_names (list): List of the feature names that have been scaled.
        session (Snowflake session): The Snowflake session for writing data.

        Returns:
        DataFrame: The original DataFrame with an additional column 'cluster_label' that contains the cluster assignment
        for each customer, along with the 'LAST_UPDATED_DATE' column.
        """

        # Loop through each segment in the `self.best_k`
        for idx, row in self.best_k.iterrows():
            segment = row['SEGMENT']
            best_k = row['best_k_silhouette']

            if idx == 1 or idx % 100 == 0:
                print(f"[{idx}] Clustering Segment: {segment} with {best_k} clusters")

            # Filter the data for the current segment
            segment_df = self.df_scaled_parts[self.df_scaled_parts["SEGMENT"] == segment]

            # Extract the features to be used for clustering
            X = segment_df[self.scaled_feature_names].dropna()  # Remove rows with missing values

            # Initialize KMeans with the best K for this segment
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

            # Fit the model and predict the clusters for the current segment
            labels = kmeans.fit_predict(X)

            # Assign the cluster labels to the original dataframe
            self.df_scaled_parts.loc[segment_df.index, "cluster_label"] = labels

        # Add the 'LAST_UPDATED_DATE' column to the dataframe
        # self.df_scaled_parts['LAST_UPDATED_DATE'] = last_updated_date
        
        self.df_scaled_parts["RUNID"] = self.runid
        self.df_scaled_parts["RUNTIMESTAMP"] = self.runtimestamp
        # Convert the Pandas DataFrame to a Snowpark DataFrame
        clusters_df = self.session.create_dataframe(self.df_scaled_parts)
        self.df_scaled_parts = None
        self.best_k = None
     
        rename_dict = {
        '"index"': 'INDEX',
        '"recency_cv"': 'RECENCY_CV',
        '"frequency_cv"': 'FREQUENCY_CV',
        '"aov_cv"': 'AOV_CV',
        '"customer_age_cv"': 'CUSTOMER_AGE_CV',
        '"customer_count"': 'CUSTOMER_COUNT',
        '"avg_cv"': 'AVG_CV',
        '"should_cluster"': 'SHOULD_CLUSTER',
        '"cluster_label"': 'CLUSTER_LABEL'}

        # Rename incorrectly quoted columns
        for old_col, new_col in rename_dict.items():
            clusters_df = clusters_df.with_column_renamed(old_col, new_col)

        # Add the CUSTOMER_SEGMENT column if not already present
        self.clusters_df = clusters_df.with_column(
            "CUSTOMER_SEGMENT",
            sql_expr("SEGMENT || '_' || CLUSTER_LABEL")
        )

        # Get 5 distinct CUSTOMER_SEGMENT values
        # clusters_df.select("CUSTOMER_SEGMENT").distinct().limit(5).show()

    def store_clusters_df(self, table):
        self.clusters_df.write.mode("append").save_as_table(table)
        
    def join_cluster_data(self):
        # Select only CUSTOMER_ID and CUSTOMER_SEGMENT from clusters_df
        clusters_segment_df = self.clusters_df.select("CUSTOMER_ID", "CUSTOMER_SEGMENT").distinct()

        # Join to sow_df
        sow_df_with_segment = self.data.join(clusters_segment_df, on="CUSTOMER_ID", how="left")
        sow_df_with_segment.select("CUSTOMER_ID", "CUSTOMER_SEGMENT").distinct().show(10)

        # If CUSTOMER_SEGMENT is null, set it to the value in SEGMENT
        sow_df_with_default_segment = sow_df_with_segment.with_column(
            "CUSTOMER_SEGMENT",
            coalesce(F_col("CUSTOMER_SEGMENT"), F_col("SEGMENT")))

        # Verify the result
        sow_df_with_default_segment.select("CUSTOMER_ID", "CUSTOMER_SEGMENT").distinct().show(10)

        before_count = (
            sow_df_with_default_segment
            .select("CUSTOMER_ID")
            .distinct()
            .count())
        
        print(f"Number of unique customers before filtering: {before_count}")

        # Step A: Find segments with at least 30 customers
        segments_with_enough_customers = (
            sow_df_with_default_segment
            .group_by("CUSTOMER_SEGMENT")
            .agg(count_distinct("CUSTOMER_ID").alias("NUM_CUSTOMERS"))
            .filter(F_col("NUM_CUSTOMERS") >= 30))

        # Step B: Filter original dataset to include only those segments
        self.sow_filtered_df = sow_df_with_default_segment.join(
            segments_with_enough_customers.select("CUSTOMER_SEGMENT"),
            on="CUSTOMER_SEGMENT",
            how="inner")

        after_count = (
            self.sow_filtered_df
            .select("CUSTOMER_ID")
            .distinct()
            .count())

        print(f"Number of unique customers after filtering: {after_count}")
        
    def save_cluster_results(self, table):
        # Write the resulting DataFrame to a Snowflake table
        self.sow_filtered_df.write.mode("append").save_as_table(table)
    
    def temp_cluster_results(self, table):
        self.sow_filtered_df.createOrReplaceTempView(table)

    def run_clustering(self, optimal_table="SOW_OPTIMAL_K_RESULTS_NTBTEST", results_table="SOW_CUSTOMER_LEVEL_CLUSTERS_NTBTEST", cluster_table="SOW_CLUSTERS_DATA_NTBTEST"):
        self.filter_df()
        self.compute_cv_summary()
        self.create_cluster_df()
        self.get_scaled_feature_names()
        self.find_optimal_k_using_silhouette()
        self.save_optimal_k(optimal_table)
        # self.temp_optimal_k(optimal_table)
        self.assign_clusters_to_customers()
        self.store_clusters_df(cluster_table)
        self.join_cluster_data()
        self.save_cluster_results(results_table)
        # self.temp_cluster_results(results_table)