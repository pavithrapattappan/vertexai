# Import necessary functions
from snowflake.snowpark.functions import (when, col as F_col, lit, lower, ltrim, to_date, to_char, current_date, datediff,
                                          coalesce, sum as F_sum, min as F_min, max as F_max, sql_expr, regexp_extract as F_regexp_extract, year, count_distinct,
                                          lag, avg as F_avg, round as sf_round, row_number)
from functools import reduce
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import StringType  

class create_data():

    def __init__(self, session, runid, runtimestamp, valid_industries=["INDUSTRIAL", "CONSUMER", "PUBLIC SECTOR", "MULTIFAMILY", "COMMERCIAL", "HOSPITALITY", "HEALTHCARE"]):
        self.session = session
        self.valid_industries = valid_industries
        self.runid = runid
        self.runtimestamp = runtimestamp

    def get_carn(self):
        """
        Load customer account and rep metadata from the CARN table.
        This includes customer location, lifecycle, and sales org structure.
        """
        self.carn_df = self.session.table("DM_SALES_MARKETING.PRODUCTION.COMBINED_ACCOUNTS_REPS_NAMS").select(
            F_col("SRC").alias("MOST_RECENT_SRC"),
            F_col("CUSTOMER_ID").alias("CUSTOMER_ID__"),
            F_col("CHANNEL").alias("MOST_RECENT_CHANNEL"),
            F_col("INDUSTRY_CODE_DESC").alias("MOST_RECENT_INDUSTRY_CODE_DESC"),
            F_col("COMPANY_TYPE_DESC").alias("MOST_RECENT_COMPANY_TYPE_DESC"),
            "CUSTOMER_NAME",
            "WINNING_RELATIONSHIP_ID",
            "WINNING_RELATIONSHIP_NAME",
            "TOPLINK_ID",
            "TOPLINK_NAME",
            "MSA",
            "MSA_DESC",
            "MSA_REGION",
            "STREET",
            "CITY",
            "STATE",
            "CUSTOMER_LIFE_CYCLE_DESC",
            "APEX_ID",
            "APEX_NAME",
            "APEX_TYPE",
            "NATIONAL_ACCOUNT_FLAG",
            "SALESFORCE_UNIQUE_ID",
            "FIRST_SALE_DATE",
            "SALES_REP_NAME",
            "REGION_MANAGER_NAME",
            "AREA_MANAGER_NAME",
            "SALES_REP_BP_ID",
            "CUSTOMER_SHIP_TO_CHAIN_ID",
            "CUSTOMER_SHIP_TO_CHAIN_NAME"
        ).filter(F_col("INDUSTRY_CODE_DESC").isin(self.valid_industries)).distinct()

    def partition(self):
        """
        Define Window to get earliest order date per customer_id and order_number
        """
        self.order_window = Window.partition_by("ORDER_NUMBER")

    def get_sales(self):
        """
        Loads and cleans transactional sales data from the combined HDS/HDPro dataset.

        Filters out:
        - Invalid order numbers
        - Unwanted sales doc types (e.g., credits/returns)
        - 'DO NOT USE' product categories
        - Specific non-revenue channels
        - Placeholder or test orders
        - Internal/invalid company types

        Adds:
        - Parsed order date
        - Harmonized material ID and description across SRCs
        - Aggregated billing quantity, GSAR, COGS, and margin at order-material level
        """
        #  Identify order_numbers with more than one customer_id
        multi_customer_orders = (
            self.session.table("DM_SALES_MARKETING.PRODUCTION.COMBINED_HDSHDP_SALES_WITH_CD1")
            .filter(F_col("CUSTOMER_ID").is_not_null() & F_col("ORDER_NUMBER").is_not_null())
            .group_by("ORDER_NUMBER")
            .agg(count_distinct(F_col("CUSTOMER_ID")).alias("n_customers"))
            .filter(F_col("n_customers") > 1)
            .select("ORDER_NUMBER")
        )
        
        self.sales_df = (
            self.session.table("DM_SALES_MARKETING.PRODUCTION.COMBINED_HDSHDP_SALES_WITH_CD1")
          	.join(multi_customer_orders, on="ORDER_NUMBER", how="left_anti")  # Exclude violating orders
            .filter(
                (F_col("SRC").isin(["HDS", "HDPro"])) &
                (F_col("CUSTOMER_ID").is_not_null()) &
                (F_col("ORDER_NUMBER").is_not_null()) & (F_col("ORDER_NUMBER") != lit("0")) &
                (~F_col("SALES_DOC_TYPE").isin(["ZCR", "ZRE"])) &
                (~F_col("PCAT").isin(["DO NOT USE"])) &
                (~F_col("MCAT").isin(["DO NOT USE"])) &
                (~F_col("WW_CHANNEL").isin(["RENOVATION", "INSTALLATION"])) &
                (~F_col("ORDER_NUMBER").like("P%")) &
                (~lower(F_col("COMPANY_TYPE_DESC")).like("%do not use%")) &
                (F_col("INDUSTRY_CODE_DESC").isin(self.valid_industries)) 
            )
            .with_column(
                "ORDER_DATE",
                F_min(to_date(F_col("DATE_DT_INT").cast("string"), "YYYYMMDD")).over(self.order_window)
            )
            # Harmonize material ID across SRCs
            .with_column(
                "MATERIAL_ID",
                when(F_col("SRC") == "HDS", F_col("HDS_MATERIAL_ID")).otherwise(F_col("HDP_SKU_ID"))
            )
            # Harmonize material descriptions across SRCs
            .with_column(
                "MATERIAL_DESC",
                when(F_col("SRC") == "HDS", F_col("MATERIAL_DESC")).otherwise(F_col("HDP_SKU_USN_DESCRIPTION"))
            )
            # Aggregate sales values at the material level per order
            .group_by(
                "SRC", "INDUSTRY_CODE_DESC", "COMPANY_TYPE_DESC", "CUSTOMER_ID",
                "ORDER_NUMBER", "ORDER_DATE", "MCAT_ID", "MCAT",
                "PCAT_ID", "PCAT", "MATERIAL_ID", "MATERIAL_DESC"
            )
            .agg(
                F_sum("BILL_QTY").alias("BILL_QTY"),
                F_sum("GSAR").alias("GSAR"),
                F_sum("COST").alias("COGS"),
                F_sum("PART_MARGIN").alias("PART_MARGIN")
            )
        ).distinct()

    def get_attributions(self):
        """
        Get attibution data
        """
        self.attr_df = (
            self.session.table("DM_SALES_MARKETING.MARKETINGOPS.ENTERPRISE_ORDER_ATTRIBUTIONS")
            .select(
                "CUSTOMER_NUMBER",
                "ORDER_ID",
                "PURCHASE_DOC_TYPE",
                "TOS_PLATFORM",
                "TOS_CUSTOMER_TYPE"
            )
            .distinct()
        )

    def carn_sales_join(self): 
        """
        Enriches sales data with:
        1. Enterprise order attribution details (TOS platform, purchase type, customer type)
        2. Core customer metadata from the CARN dataset (account and rep-level fields)
        """
        attr_sales_df = self.sales_df.join(
            self.attr_df,
            (self.sales_df["CUSTOMER_ID"] == self.attr_df["CUSTOMER_NUMBER"]) &
            (self.sales_df["ORDER_NUMBER"] == self.attr_df["ORDER_ID"]),
            how="left"
        )
        # Now join with CARN data
        self.enriched_df = attr_sales_df.join(
                self.carn_df,
                (attr_sales_df["CUSTOMER_ID"] == self.carn_df["CUSTOMER_ID__"]),
                how="left"
            )
        self.carn_df = None
        self.sales_df = None
        self.attr_df = None
        
    def exclude_customers(self):
        """
        Removes:
        1. Customers with closed lifecycle statuses (from ENTERPRISE_CUSTOMER_NEW)
        2. Multifamily customers with 'rehab', 'reno', or 'rhb' in CUSTOMER_NAME

        Returns:
            Cleaned Snowpark DataFrame
        """
        total_before = self.enriched_df.select("CUSTOMER_ID").distinct().count()

        # -- A. CLOSED CUSTOMERS -------------------------------------
        closed_statuses = [
            "ACCOUNT CLOSED", "INTERNAL MIGRATION AND CLOSED", "ACCOUNT MIGRATED",
            "ACCOUNT MIGRATED LIMITED ACCESS", "ACCOUNT MIGRATED ALLOW RETURNS",
            "ACCOUNT MIGRATED AND CLOSED", "Closed Customer - Customer Requested",
            "Closed Customer - Duplicate", "Closed Customer - Derogatory"
        ]

        closed_ids_df = (
            self.session.table("DM_SALES_MARKETING.MARKETINGOPS_DW.ENTERPRISE_CUSTOMER_NEW")
            .filter(F_col("CUSTOMER_LIFE_CYCLE_DESC").isin(closed_statuses))
            .select("CUSTOMER_ID")
            .distinct()
        )

        num_closed = closed_ids_df.select("CUSTOMER_ID").distinct().count()

        print(f"Closed customers to exclude: {num_closed:,}")

        # -- B. REHAB MULTIFAMILY CUSTOMERS --------------------------
        rehab_condition = (
            (F_col("INDUSTRY_CODE_DESC") == "MULTIFAMILY") &
            (
                lower(F_col("CUSTOMER_NAME")).like("%rehab%") |
                lower(F_col("CUSTOMER_NAME")).like("%reno%") |
                lower(F_col("CUSTOMER_NAME")).like("%rhb%")
            )
        )

        rehab_ids_df = (
            self.enriched_df.filter(rehab_condition)
            .select("CUSTOMER_ID")
            .distinct()
        )

        num_rehab = rehab_ids_df.select("CUSTOMER_ID").distinct().count()

        print(f"Rehab customers to exclude: {num_rehab:,}")

        # -- C. UNION ALL EXCLUSIONS AND REMOVE ---------------------
        exclusion_ids = closed_ids_df.union_all(rehab_ids_df).distinct()

        self.df_clean = self.enriched_df.join(exclusion_ids, on="CUSTOMER_ID", how="left_anti")
        self.enriched_df = None

        total_after = self.df_clean.select("CUSTOMER_ID").distinct().count()
        print(f"Total distinct customers BEFORE exclusion: {total_before:,}")
        print(f"Total distinct customers AFTER exclusion: {total_after:,}")
        print(f"Customers removed: {total_before - total_after:,}\n")
 

    def get_agreement(self):
        """
        Load agreement table and add UNDER_CONTRACT flag
        """
        self.agreement_df = (
            self.session.table("dm_sales_marketing.development_qa.iih_active_agreement")
            .select(
                F_col("ACCOUNT_ID"),
                when(
                    F_col("AGREEMENT_END_DATE") >= current_date(), lit("Y"))
                    .otherwise(lit("N"))
                    .alias("UNDER_CONTRACT")
            )
            .distinct()
        )

    def build_final(self):
        """
        Adds a CONTRACT_STATUS column to indicate if a customer is under an active agreement.

        Logic:
        - Flags customers as 'Y' if their agreement end date is in the future.
        - Joins on CUSTOMER_ID (after trimming leading zeros) to map to ACCOUNT_ID.
        """
        self.final_df = (
            self.df_clean.join(
                self.agreement_df,
                ltrim(self.df_clean["CUSTOMER_ID"], lit("0")) == self.agreement_df["ACCOUNT_ID"],
                how="left"
            )
            .with_column("CONTRACT_STATUS", coalesce(F_col("UNDER_CONTRACT"), lit("N")))
        )
        self.df_clean = None
        self.agreement_df = None

    def build_combined_tam(self):
        """
        Enriches customer data with TAM (Total Addressable Market), Unit Counts,
        and Building Age by combining sources: Healthcare, Multifamily, and Housing Authority.

        Logic:
        - Consolidates three TAM reference datasets (HOSP, MH, HA)
        - Merges with master TAM mapping to obtain CUSTOMER_ID
        - Selects best available TAM/unit values and computes building age
        """

        self.combined_tam_df = (self.session.table("DM_SALES_MARKETING.PRODUCTION.TAM_MH_HS_HL_PS_FINAL_2025").with_column_renamed("LOCATIONGROUPID", "LOCATION_GROUP_ID").select(
        F_col("CUSTOMER_ID"),
        F_col("LOCATION_GROUP_ID"),
        F_col("FINAL_TAM"),
        F_col("UNITS").alias("FINAL_UNITS"),
        F_col("YEAR_BUILT")
        )
    )

    
    def get_master_tam(self):
        """
        Load the master TAM table
        """
        self.master_tam_df = (self.session.table("DM_SALES_MARKETING.PRODUCTION.COMBINED_TAM").select(
            "LOCATION_GROUP_ID","CUSTOMER_ID","TAM_ID","UNITS","TAM").with_column("LOCATION_GROUP_ID_CLEANED", F_regexp_extract(F_col("LOCATION_GROUP_ID"), r"^(\d+)", 1)))

    def clean_values(self):
        """
        Rename columns in combined_tam_df to align with master table naming
        """
        tam_df = (
            self.master_tam_df.join(
            self.combined_tam_df,
            on="CUSTOMER_ID",
            how="full"))
        self.master_tam_df = None
        self.combined_tam_df = None
                                                                                  
        self.tam_df = tam_df.select(
            F_col("CUSTOMER_ID"),
            F_col("TAM_ID"),
            coalesce(F_col("FINAL_UNITS"), F_col("UNITS")).alias("UNITS"),
            when(
                (F_col("FINAL_TAM").is_not_null()) & (F_col("FINAL_TAM") > 0),
                F_col("FINAL_TAM")
            ).when(
                (F_col("TAM").is_not_null()) & (F_col("TAM") > 0),
                F_col("TAM")
            ).otherwise(lit(None)).alias("TAM"),

            F_col("YEAR_BUILT"),
    
            when(
                F_col("YEAR_BUILT").is_not_null(),
                year(current_date()) - F_col("YEAR_BUILT").cast("int")
            ).alias("BUILDING_AGE_YEARS")
        )
                
                                                                                   

    def get_sale_date(self):
        """
        Computes the earliest known sale date for each customer by:
        - Taking min(FIRST_SALE_DATE) from CARN (if available)
        - Taking min(ORDER_DATE) from sales
        - Applying fallback logic to pick the better of the two
        Returns:
            DataFrame with CUSTOMER_ID and FIRST_SALE_DATE
        """
        self.date_df = (
            self.final_df.group_by("CUSTOMER_ID")
                .agg(
                    F_min("FIRST_SALE_DATE").alias("CARN_FIRST_SALE_DATE"),
                    F_min("ORDER_DATE").alias("BILLING_MIN_DATE")
                )
                .with_column(
                    "FIRST_BILLING_DATE",
                    when(
                        F_col("CARN_FIRST_SALE_DATE").is_null(), F_col("BILLING_MIN_DATE")
                    ).when(
                        to_char(F_col("CARN_FIRST_SALE_DATE"), "YYYY") == "1900", F_col("BILLING_MIN_DATE")
                    ).when(
                        F_col("BILLING_MIN_DATE").is_null(), F_col("CARN_FIRST_SALE_DATE")
                    ).when(
                        F_col("CARN_FIRST_SALE_DATE") < F_col("BILLING_MIN_DATE"), F_col("CARN_FIRST_SALE_DATE")
                    ).otherwise(F_col("BILLING_MIN_DATE"))
                )
                .select("CUSTOMER_ID", "FIRST_BILLING_DATE").distinct()
            )
        self.date_df.select("CUSTOMER_ID", "FIRST_BILLING_DATE").distinct()
        
    def get_tiers(self):
        """
        Retrieves National Account Tier assignments for both HDS and HDPro customers.

        Logic:
        - For HDS: Extract CUSTOMER_ID from MC_CUSTOMER_ID
        - For HDPro: Extract CUSTOMER_ID from MC_CUSTOMER_SHIP_TO_LEGACY_ID
        - Filters only 'National Account' records
        - Returns CUSTOMER_ID to TIER mapping
        """
        hds_tiers = (
            self.session.table("INTEGRATION.FINANCE.XREF_CUSTOMER_TIER")
            .filter(
                (F_col("MC_NATIONAL_ACCOUNT_FLAG") == "National Account") &
                (sql_expr("SPLIT_PART(MC_CUSTOMER_ID, '|', 1)") == "HDS")
            )
        ).distinct()

        # -- Load and filter HDPro National Account tier data ----------
        hdp_tiers = (
            self.session.table("INTEGRATION.FINANCE.XREF_CUSTOMER_TIER")
            .filter(
                (F_col("MC_NATIONAL_ACCOUNT_FLAG") == "National Account") &
                (sql_expr("SPLIT_PART(MC_CUSTOMER_SHIP_TO_LEGACY_ID, '|', 1)") == "PRO")
            )
        ).distinct()

        # -- Extract CUSTOMER_ID and assign TIER for both sources ------
        hds_tier_df = hds_tiers.select(
            sql_expr("SPLIT_PART(MC_CUSTOMER_ID, '|', 2)").alias("CUSTOMER_ID"),
            F_col("TIER_ASSIGN").alias("TIER")
        ).distinct()

        hdp_tier_df = hdp_tiers.select(
            sql_expr("SPLIT_PART(MC_CUSTOMER_SHIP_TO_LEGACY_ID, '|', 2)").alias("CUSTOMER_ID"),
            F_col("TIER_ASSIGN").alias("TIER")
        ).distinct()

        # -- Combine and deduplicate tier assignments ------------------
        tier_df = hds_tier_df.union_all(hdp_tier_df).distinct()
        self.tier_df = tier_df.select("CUSTOMER_ID", "TIER")

    def add_stamps(self):
        """
        Enriches the main customer-level DataFrame with metadata including:
        - Contract status
        - TAM, Units, Building Age
        - First sale date and customer age
        - National account tier

        Joins are performed on CUSTOMER_ID.
        """
        
        self.stamp_df = self.final_df.join(self.tam_df, on="CUSTOMER_ID", how="left") \
            .join(self.date_df, on="CUSTOMER_ID", how="left") \
            .join(self.tier_df, on="CUSTOMER_ID", how="left") \
            .with_column("RUNID", lit(self.runid)) \
            .with_column("RUNTIMESTAMP", lit(self.runtimestamp))
        self.final_df = None
        self.tam_df = None
        self.date_df = None
        self.tier_df = None

    def write_history(self, table):
        self.stamp_df.write.mode("append").save_as_table(table)
                                                                                  
    def temp_history(self, table):
        self.stamp_df.createOrReplaceTempView(table)
                     
                     
    ### 
    ### START FEATURE ENGINEERING NOTEBOOK
    ###
    def compute_behavioral_metrics(self):
        """
        Compute key behavioral metrics:
        - Recency (days since last order)
        - Average days between orders
        - Average order value (GSAR / orders)
        - Order frequency (orders per year)
        """
        print("Starting behavioral metrics computation...")
        
        orders_missing_date = (self.stamp_df.filter(F_col("ORDER_DATE").is_null())
                               .select("ORDER_NUMBER").distinct().count())
                               
        print("Number of orders with missing ORDER_DATE:", orders_missing_date)
              
        orders_df = (self.stamp_df.filter(F_col("ORDER_DATE").is_not_null())
        .group_by("CUSTOMER_ID","ORDER_NUMBER", to_date(F_col("ORDER_DATE")).alias("ORDER_DATE"))
                  .agg(F_sum("GSAR").alias("GSAR"))).distinct()

        print(f"Order rows prepared: {orders_df.count():,}")

        # Recency
        recency_df = orders_df.group_by("CUSTOMER_ID").agg(
            datediff("day", F_max("ORDER_DATE"), current_date()).alias("RECENCY_DAYS")).distinct()
        # print("recency_df created")

        # Average days between orders
        order_window = Window.partition_by("CUSTOMER_ID").order_by("ORDER_DATE")
        # print("order_window")
        orders_with_lag = orders_df.with_column("PREV_ORDER_DATE", lag("ORDER_DATE").over(order_window)) \
            .with_column("DAYS_BETWEEN_ORDERS", datediff("day", F_col("PREV_ORDER_DATE"), F_col("ORDER_DATE")))
        # print("order_with_lag created")

        avg_days_df = orders_with_lag.filter(
            F_col("DAYS_BETWEEN_ORDERS").is_not_null() & (F_col("DAYS_BETWEEN_ORDERS") >= 0)
        ).group_by("CUSTOMER_ID").agg(
            F_avg("DAYS_BETWEEN_ORDERS").alias("AVG_DAYS_BETWEEN_ORDERS")).distinct()
        # print("avg_days_df created")

        # Average Order Value
        aov_df = orders_df.group_by("CUSTOMER_ID").agg(
            (F_sum("GSAR") / count_distinct("ORDER_NUMBER")).alias("AVG_ORDER_VALUE")
        ).distinct()
        # print("aov_df created")

        # Order Frequency
        total_orders_df = orders_df.group_by("CUSTOMER_ID").agg(
            count_distinct("ORDER_NUMBER").alias("TOTAL_ORDERS")
        ).distinct()
        # print("total_orders_df created")

        billing_min_dates = orders_df.group_by("CUSTOMER_ID").agg(
            F_min("ORDER_DATE").alias("BILLING_MIN_DATE")
        ).distinct()
        # print("billing_min_dates created")

        customer_tenure_df = billing_min_dates.select(
            "CUSTOMER_ID",
            (datediff("day", F_col("BILLING_MIN_DATE"), current_date()) / lit(365.0)).cast("double").alias("CUSTOMER_TENURE_YEARS")
        ).distinct()
        # print("customer_tenure_df created")

        order_frequency_df = total_orders_df.join(customer_tenure_df, on="CUSTOMER_ID", how="left") \
            .with_column("ORDER_FREQUENCY",
                         when(F_col("CUSTOMER_TENURE_YEARS") > 0, (F_col("TOTAL_ORDERS") / F_col("CUSTOMER_TENURE_YEARS")).cast("double"))
                         .otherwise(None)
                        ).distinct()
        # print("order_frequency_df created")

        self.metrics_df = reduce(lambda l, r: l.join(r, on="CUSTOMER_ID", how="left"), [recency_df, avg_days_df, aov_df, order_frequency_df])
        # print("metrics_df created")
        
        self.behav_df = self.stamp_df.join(self.metrics_df, on="CUSTOMER_ID", how="left")
        # print("Behavioral metrics computed")
        self.metrics_df = None
        self.stamp_df = None
    
    def compute_customer_age(self):
        """Compute customer age from first billing date."""
        self.age_df = self.behav_df.with_column(
            "CUSTOMER_AGE_YEARS", sf_round(datediff("day", F_col("FIRST_BILLING_DATE"), current_date()) / lit(365), 0))
        self.behav_df = None
    
    def consolidate_tam_values(self):
        """Replace TAM with maximum value when duplicates exist for a customer."""
        tam_check_df = self.age_df.group_by("CUSTOMER_ID").agg(count_distinct("TAM").alias("NUM_UNIQUE_TAM")).filter(F_col("NUM_UNIQUE_TAM") > 1)
        
        max_tam_df = self.age_df.join(tam_check_df.select("CUSTOMER_ID"), on="CUSTOMER_ID", how="inner") \
            .group_by("CUSTOMER_ID").agg(F_max("TAM").alias("MAX_TAM"))

        self.consolidate_df = self.age_df.join(max_tam_df, on="CUSTOMER_ID", how="left") \
            .with_column("TAM", when(F_col("MAX_TAM").is_not_null(), F_col("MAX_TAM")).otherwise(F_col("TAM"))).drop("MAX_TAM")
        self.age_df = None
        
        # print("Consolidated TAM")

    def derive_most_recent_channel(self):
        """Tag each customer with their most recent purchase channel."""
        channel_window = Window.partition_by("CUSTOMER_ID").order_by(F_col("ORDER_DATE").desc())
        channel_ranked_df = self.consolidate_df.with_column("ROW_NUM", row_number().over(channel_window)) \
            .filter(F_col("ROW_NUM") == 1).select("CUSTOMER_ID", F_col("MOST_RECENT_CHANNEL").alias("MOST_RECENT_CHANNEL"))

        self.consolidate_df.drop("MOST_RECENT_CHANNEL").join(channel_ranked_df, on="CUSTOMER_ID", how="left").with_column_renamed("MOST_RECENT_CHANNEL", "MOST_RECENT_CHANNEL")
        
        # print("Derived most recent channel")

    def format_columns(self):
        """Standardize column formats for presentation and storage."""
        self.consolidate_df \
            .with_column("GSAR", sf_round(F_col("GSAR"), 2)) \
            .with_column("COGS", sf_round(F_col("COGS"), 2)) \
            .with_column("PART_MARGIN", sf_round(F_col("PART_MARGIN"), 2)) \
            .with_column("TAM", sf_round(F_col("TAM"), 2)) \
            .with_column("AVG_ORDER_VALUE", sf_round(F_col("AVG_ORDER_VALUE"), 2)) \
            .with_column("BILL_QTY", sf_round(F_col("BILL_QTY"), 0)) \
            .with_column("CUSTOMER_UNITS", sf_round(F_col("UNITS"), 0)).drop('UNITS') \
            .with_column("YEAR_BUILT", sf_round(F_col("YEAR_BUILT"), 0)) \
            .with_column("ORDER_FREQUENCY", sf_round(F_col("ORDER_FREQUENCY"), 2)) \
            .with_column("AVG_DAYS_BETWEEN_ORDERS", sf_round(F_col("AVG_DAYS_BETWEEN_ORDERS"), 0)) \
            .with_column("CHANNEL", coalesce(F_col("CHANNEL"), lit("Unassigned")))
            # .with_column("RUNID", lit(runid)) \
            # .with_column("RUNTIMESTAMP", lit(runtimestamp))
            
        # print("Formated columns")

    def write_data_features(self, table):
        print("Creating table...")
        self.consolidate_df.write.mode("append").save_as_table(table)
                                                                                  
    def temp_features(self, table):
        print("Creating temp table...")
        self.consolidate_df.createOrReplaceTempView(table)
    
    def extract_and_join(self, h_table, f_table):
        self.get_carn()
        self.partition()
        self.get_sales()
        self.get_attributions()
        self.carn_sales_join()
        self.exclude_customers()
        self.get_agreement()
        self.build_final()
        self.build_combined_tam()
        self.get_master_tam()
        self.clean_values()
        self.get_sale_date()
        self.get_tiers()
        self.add_stamps()
        self.write_history(h_table) # history table param
        # self.temp_history(h_table)  # temp history table param
        # print("Start feature eng")
        
        ### Feature engineering
        self.compute_behavioral_metrics()
        self.compute_customer_age()
        self.consolidate_tam_values()
        self.derive_most_recent_channel()
        self.format_columns()                                
        self.write_data_features(f_table) # features table param
        # self.temp_features(f_table) # temp features taable param
        # print("Finished feature eng")