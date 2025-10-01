# Import necessary functions
import pandas as pd 
import numpy as np

import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from builtins import abs

from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import (col as F_col, to_date, count_distinct, approx_percentile,
                                          avg as F_avg, sum as F_sum, min as F_min, max as F_max, year, 
                                          when, lit, lag, round as sf_round, coalesce, expr as sql_expr)
from sklearn.preprocessing import LabelEncoder, RobustScaler

class preprocess_data():
    def __init__(self, data):
        self.data = data

    def calc_spend_tiers(self):
        self.data_order = self.data.with_column("ORDER_YEAR", year(F_col("ORDER_DATE")))
        
        annual_gsar_df = (
            self.data_order.group_by("CUSTOMER_ID", "ORDER_YEAR")
            .agg(F_sum("GSAR").alias("YEARLY_GSAR")))
        
        avg_annual_df = (
            annual_gsar_df.group_by("CUSTOMER_ID")
                        .agg(F_avg("YEARLY_GSAR").alias("AVG_ANNUAL_SPEND")))

        self.data_annual = self.data_order.join(avg_annual_df, on="CUSTOMER_ID", how="left")

        tier_map_expr = (
            when(F_col("TIER") == "Tier 0", lit(0))
            .when(F_col("TIER") == "Tier 1", lit(1))
            .when(F_col("TIER") == "Tier 2", lit(2))
            .when(F_col("TIER") == "Tier 3", lit(3))
            .when(F_col("TIER") == "Other", lit(4))
        )

        self.data_tiers = self.data_annual.with_column("TIER_NUMERIC", tier_map_expr)

        self.data_tiers_tam = self.data_tiers.with_column(
            "TAM_TIER_OR_AVGSPEND",
            when(F_col("TAM").is_not_null(), F_col("TAM"))
            .when(F_col("TIER_NUMERIC").is_not_null(), F_col("TIER_NUMERIC"))
            .otherwise(F_col("AVG_ANNUAL_SPEND"))
        )

        self.data_source = self.data_tiers_tam.with_column(
            "TAM_TIER_SOURCE",
            when(F_col("TAM").is_not_null(), lit("TAM"))
            .when(F_col("TIER_NUMERIC").is_not_null(), lit("TIER"))
            .when(F_col("AVG_ANNUAL_SPEND").is_not_null(), lit("AVG_ANNUAL_SPEND"))
            .otherwise(lit("UNKNOWN"))
        )

        distinct_customers = self.data_source.select("CUSTOMER_ID", "INDUSTRY_CODE_DESC", "TAM_TIER_OR_AVGSPEND").distinct()

        industry_counts = (
            distinct_customers.group_by("INDUSTRY_CODE_DESC")
            .agg(count_distinct("CUSTOMER_ID").alias("CUSTOMER_COUNT"))
        )

        null_counts = (
            distinct_customers.group_by("INDUSTRY_CODE_DESC")
            .agg(
                count_distinct(
                    when(F_col("TAM_TIER_OR_AVGSPEND").is_null(), F_col("CUSTOMER_ID"))
                ).alias("NULL_CUSTOMERS")
            )
        )

        null_percentage = (
            null_counts.join(industry_counts, on="INDUSTRY_CODE_DESC")
            .with_column(
                "NULL_PCT",
                sf_round(F_col("NULL_CUSTOMERS") * 100 / F_col("CUSTOMER_COUNT"), 2)
            )
            .select("INDUSTRY_CODE_DESC", "CUSTOMER_COUNT", "NULL_PCT")
            .order_by("INDUSTRY_CODE_DESC")
        )

        # print("Percent Null Fallback Values by Industry: ", null_percentage.show())

        # print("Customers with ALL fallback values NULL:")
        # self.data_source_filter = self.data_source.filter(
        #     F_col("TAM").is_null() &
        #     F_col("TIER_NUMERIC").is_null() &
        #     F_col("AVG_ANNUAL_SPEND").is_null()
        # ).select("CUSTOMER_ID", "INDUSTRY_CODE_DESC").distinct().show()

        binning_sources = ["TAM", "AVG_ANNUAL_SPEND"]

        thresholds_df = (
            self.data_source.filter(
                F_col("TAM_TIER_SOURCE").isin(binning_sources) &
                F_col("TAM_TIER_OR_AVGSPEND").is_not_null()
            )
            .select("INDUSTRY_CODE_DESC", "TAM_TIER_SOURCE", "CUSTOMER_ID", "TAM_TIER_OR_AVGSPEND")
            .distinct()
            .group_by("INDUSTRY_CODE_DESC", "TAM_TIER_SOURCE")
            .agg(
                approx_percentile("TAM_TIER_OR_AVGSPEND", lit(0.25)).alias("value_q1"),
                approx_percentile("TAM_TIER_OR_AVGSPEND", lit(0.50)).alias("value_q2"),
                approx_percentile("TAM_TIER_OR_AVGSPEND", lit(0.75)).alias("value_q3")
            )
        )

        self.data_thre = self.data_source.join(thresholds_df, on=["INDUSTRY_CODE_DESC", "TAM_TIER_SOURCE"], how="left")

        self.data_bin = self.data_thre.with_column(
            "TAM_TIER_BIN",
            when(F_col("TAM_TIER_SOURCE") == "TIER", F_col("TIER"))
            .when(F_col("TAM_TIER_OR_AVGSPEND").is_null(), lit("NA"))
            .when(F_col("TAM_TIER_OR_AVGSPEND") <= F_col("value_q1"), lit("Low"))
            .when((F_col("TAM_TIER_OR_AVGSPEND") > F_col("value_q1")) & (F_col("TAM_TIER_OR_AVGSPEND") <= F_col("value_q2")), lit("Medium-Low"))
            .when((F_col("TAM_TIER_OR_AVGSPEND") > F_col("value_q2")) & (F_col("TAM_TIER_OR_AVGSPEND") <= F_col("value_q3")), lit("Medium-High"))
            .otherwise(lit("High"))
        )
        # print("Tier bins created")
#################
    def create_segment(self):
        self.data_seg = self.data_bin.with_column(
            "RULE_BASED_SEGMENT",
            sql_expr(
                "COALESCE(INDUSTRY_CODE_DESC, 'NA') || '_' || "
                "COALESCE(COMPANY_TYPE_DESC, 'NA') || '_' || "
                "COALESCE(MSA_REGION, 'NA') || '_' || "
                "COALESCE(MSA, 'NA') || '_' || "
                "COALESCE(TAM_TIER_SOURCE, 'NA')|| '_' || "
                "COALESCE(TAM_TIER_BIN, 'NA')"
            )
        )

        self.data_seg_msa = self.data_seg.with_column(
            "RULE_BASED_SEGMENT_NO_MSA",
            sql_expr(
                "COALESCE(INDUSTRY_CODE_DESC, 'NA') || '_' || "
                "COALESCE(COMPANY_TYPE_DESC, 'NA') || '_' || "
                "COALESCE(MSA_REGION, 'NA') || '_' || "
                "COALESCE(TAM_TIER_SOURCE, 'NA') || '_' || "
                "COALESCE(TAM_TIER_BIN, 'NA')"
            )
        )

        self.data_seg_msa_region = self.data_seg_msa.with_column(
            "RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION",
            sql_expr(
                "COALESCE(INDUSTRY_CODE_DESC, 'NA') || '_' || "
                "COALESCE(COMPANY_TYPE_DESC, 'NA') || '_' || "
                "COALESCE(TAM_TIER_SOURCE, 'NA') || '_' || "
                "COALESCE(TAM_TIER_BIN, 'NA')"
            )
        )
        # print("created segment columns")

        seg_1 = self.data_seg_msa_region.group_by("RULE_BASED_SEGMENT").agg(count_distinct("CUSTOMER_ID").alias("COUNT_SEG_1"))
        seg_2 = self.data_seg_msa_region.group_by("RULE_BASED_SEGMENT_NO_MSA").agg(count_distinct("CUSTOMER_ID").alias("COUNT_SEG_2"))
        seg_3 = self.data_seg_msa_region.group_by("RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION").agg(count_distinct("CUSTOMER_ID").alias("COUNT_SEG_3"))

        self.data_segs = self.data_seg_msa_region.join(seg_1, on="RULE_BASED_SEGMENT", how="left")\
            .join(seg_2, on="RULE_BASED_SEGMENT_NO_MSA", how="left")\
            .join(seg_3, on="RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION", how="left")
        
        self.data_segs_cnt = self.data_segs.with_column(
            "SEGMENT",
            when(F_col("COUNT_SEG_1") > 60, F_col("RULE_BASED_SEGMENT"))
            .when(F_col("COUNT_SEG_2") > 60, F_col("RULE_BASED_SEGMENT_NO_MSA"))
            .otherwise(F_col("RULE_BASED_SEGMENT_NO_MSA_NO_MSA_REGION"))
        )
        # print("create count segment columns")
        
    def write_prep_data(self, table):
        self.data_segs_cnt.write.mode("append").save_as_table(table)
        
    def create_segment_freq(self, table):
        self.calc_spend_tiers()
        self.create_segment()
        self.write_prep_data(table)