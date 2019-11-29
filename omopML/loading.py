import pyspark.sql.functions as F
from pyspark.sql.functions import col, concat, when, first, rank, isnull, lit, coalesce, asc, desc
from pyspark.sql.window import Window

class DataLoader:
    def __init__(self, database, tables_dict, measurements=None, at_admission=False):
        self.database = database
        self.tables_list = list(tables_dict.keys())
        self.tables_dict = tables_dict
        self.at_admission = at_admission
        self.measurements = measurements
        self.loaded_tables = {}
        self.tables_to_join = {}
    
    def load(self, sqlContext):
        for table in self.tables_list:
            self.loaded_tables[table] = (
                sqlContext.sql("select * from " + self.database + ".{}".format(table))
            )
            
    def filter(self, conditions):
        if "gender_concept_id" in conditions:
            self.loaded_tables["person"] = (
                self.loaded_tables["person"]
                    .where(col("gender_concept_id") == conditions["gender_concept_id"])
            )
        if "year_of_birth_min" in conditions:
            self.loaded_tables["person"] = (
                self.loaded_tables["person"]
                    .where(col("year_of_birth") >= conditions["year_of_birth_min"])
            )
        if "year_of_birth_max" in conditions:
            self.loaded_tables["person"] = (
                self.loaded_tables["person"]
                    .where(col("year_of_birth") <= conditions["year_of_birth_max"])
            )
        if "visit_start_date" in conditions:
            self.loaded_tables["visit_detail"] = (
                self.loaded_tables["visit_detail"]
                    .where(col("visit_start_date") >= conditions["visit_start_date"])
            )
        if "visit_end_date" in conditions:
            self.loaded_tables["visit_detail"] = (
                self.loaded_tables["visit_detail"]
                    .where(col("visit_end_date") <= conditions["visit_end_date"])
            )
        if "visit_type_concept_id" in conditions:
            self.loaded_tables["visit_detail"] = (
                self.loaded_tables["visit_detail"]
                    .where(col("visit_type_concept_id") == conditions["visit_type_concept_id"])
            )
        if "visit_detail_concept_id" in conditions:
            self.loaded_tables["visit_detail"] = (
                self.loaded_tables["visit_detail"]
                    .where(col("visit_detail_concept_id").isin(conditions["visit_detail_concept_id"]))
            )
        if self.measurements:
            self.loaded_tables["measurement"] = (
                self.loaded_tables["measurement"]
                    .where(col("measurement_concept_id").isin(list(self.measurements.values())))
            )
            
    def create_dataset(self):
        for table in self.tables_dict:
            self.tables_to_join[table] = self.loaded_tables[table].select(self.tables_dict[table])
            
        self.tables_to_join["history_aids"] = (
            self.tables_to_join["note_nlp"]
            .where(
                (col("lexical_variant").rlike("(?i)history"))
                & (
                    (col("lexical_variant").rlike("(?i)aids"))
                    | (col("lexical_variant").rlike("(?i)immunodeficiency"))
                    | (col("lexical_variant").rlike("(?i)acquired immune deficiency syndrome"))
                )
                & (~col("lexical_variant").rlike("(?i)family"))
            )
            .select(col("note_id"))
            .drop_duplicates()
            .withColumn("history_aids", lit(1))
            .join(self.tables_to_join["note"], how="left", on="note_id")
            .select(
                col("person_id"),
                col("history_aids"),
                col("note_date")
            ).groupby(
                col("person_id"),
                col("history_aids")
            ).agg(
                F.min("note_date").alias("history_aids_min_date")
            )
        )
        
        self.tables_to_join["history_metastases"] = (
            self.tables_to_join["note_nlp"]
            .where(
                (col("lexical_variant").rlike("(?i)history"))
                & (
                    (col("lexical_variant").rlike("(?i)metastases"))
                    | (col("lexical_variant").rlike("(?i)metastatic"))
                    | (col("lexical_variant").rlike("(?i)metastasis"))
                    | (col("lexical_variant").rlike("(?i)secondary tumor"))
                    | (col("lexical_variant").rlike("(?i)secondary cancer"))
                )
                & (~col("lexical_variant").rlike("(?i)family"))
            )
            .select(col("note_id"))
            .drop_duplicates()
            .withColumn("history_metastases", lit(1))
            .join(self.tables_to_join["note"], how="left", on="note_id")
            .select(
                col("person_id"),
                col("history_metastases"),
                col("note_date")
            ).groupby(
                col("person_id"),
                col("history_metastases")
            ).agg(
                F.min("note_date").alias("history_metastases_min_date")
            )
        )
        
        self.tables_to_join["history_hemato"] = (
            self.tables_to_join["note_nlp"]
            .where(
                (col("lexical_variant").rlike("(?i)history"))
                & (
                    (col("lexical_variant").rlike("(?i)myeloma"))
                    | (col("lexical_variant").rlike("(?i)leukemia"))
                    | (col("lexical_variant").rlike("(?i)lymphoma"))
                )
                & (~col("lexical_variant").rlike("(?i)family"))
            )
            .select(col("note_id"))
            .drop_duplicates()
            .withColumn("history_hemato", lit(1))
            .join(self.tables_to_join["note"], how="left", on="note_id")
            .select(
                col("person_id"),
                col("history_hemato"),
                col("note_date")
            ).groupby(
                col("person_id"),
                col("history_hemato")
            ).agg(
                F.min("note_date").alias("history_hemato_min_date")
            )
        )
            
        self.tables_to_join["preceding_surgery"] = (
            self.tables_to_join["visit_detail"]
                .where(col("visit_detail_concept_id") == 4149152)
                .select(col("visit_occurrence_id"))
                .withColumn("from_surgery", lit(1))
                .drop_duplicates()
        )
        
        self.tables_to_join["icu_stays"] = (
            self.tables_to_join["visit_occurrence"]
                .join(
                    self.tables_to_join["visit_detail"],
                    how="left",
                    on="visit_occurrence_id"
                )
                .where(
                    (col("visit_detail_concept_id") == 32037)
                    & (col("visit_type_concept_id") == 2000000006)
                )
                .select(
                    col("person_id"),
                    col("visit_occurrence_id"),
                    col("visit_detail_id"),
                    col("visit_detail_concept_id"),
                    col("visit_start_date"),
                    col("visit_end_date"),
                    col("discharge_to_concept_id")
                )
            
        )
            
        window_asc = (
            Window.partitionBy(
                "person_id", 
                "visit_occurrence_id",
                "measurement_concept_id"
            )
                .orderBy(asc("measurement_datetime"))
        )
        
        window_desc = (
            Window.partitionBy(
                "person_id", 
                "visit_occurrence_id",
                "measurement_concept_id"
            )
                .orderBy(desc("measurement_datetime"))
        )
        
        self.pivot_measures_asc = (
            self.tables_to_join["icu_stays"]
            .join(
                self.tables_to_join["measurement"],
                how="left",
                on="visit_occurrence_id"
            )
            .where(
                (col("measurement_datetime") >= col("visit_start_date"))
                & (col("measurement_datetime") <= col("visit_end_date"))
            )
            .select('*', rank().over(window_asc).alias('rank'))
            .filter(col('rank') == 1)
            .groupBy("visit_occurrence_id")
            .pivot("measurement_concept_id", list(self.measurements.values()))
            .agg(first("value_as_number", ignorenulls=True))
        )
        self.pivot_measures_asc = self.rename_columns(
            self.pivot_measures_asc, {str(v): str("first_" + k) for k, v in self.measurements.items()}
        )
        
        self.pivot_measures_desc = (
            self.tables_to_join["icu_stays"]
            .join(
                self.tables_to_join["measurement"],
                how="left",
                on="visit_occurrence_id"
            )
            .where(
                (col("measurement_datetime") >= col("visit_start_date"))
                & (col("measurement_datetime") <= col("visit_end_date"))
            )
            .select('*', rank().over(window_desc).alias('rank'))
            .filter(col('rank') == 1)
            .groupBy("visit_occurrence_id")
            .pivot("measurement_concept_id", list(self.measurements.values()))
            .agg(first("value_as_number", ignorenulls=True))
        )
        self.pivot_measures_desc = self.rename_columns(
            self.pivot_measures_desc, {str(v): str("last_" + k) for k, v in self.measurements.items()}
        )
            
        self.dataset = (
            self.tables_to_join["icu_stays"]
            .join(
                self.tables_to_join["person"], 
                how="left", 
                on="person_id"
            )
            .join(
                self.tables_to_join["death"], 
                how="left",
                on="person_id"
            )
            .join(
                self.tables_to_join["preceding_surgery"], 
                how="left",
                on="visit_occurrence_id"
            )
            .withColumn("from_surgery", coalesce("from_surgery", lit(0)))
            .join(
                self.tables_to_join["history_aids"], 
                how="left",
                on="person_id"
            )
            .withColumn("history_aids", coalesce("history_aids", lit(0)))
            .join(
                self.tables_to_join["history_metastases"], 
                how="left",
                on="person_id"
            )
            .withColumn("history_metastases", coalesce("history_metastases", lit(0)))
            .join(
                self.tables_to_join["history_hemato"], 
                how="left",
                on="person_id"
            )
            .withColumn("history_hemato", coalesce("history_hemato", lit(0)))
            .join(
                self.pivot_measures_asc, 
                how="left", 
                on="visit_occurrence_id"
            )
            .join(
                self.pivot_measures_desc, 
                how="left", 
                on="visit_occurrence_id"
            )
        )
        
    def rename_columns(self, df, columns):
        if isinstance(columns, dict):
            for old_name, new_name in columns.items():
                df = df.withColumnRenamed(old_name, new_name)
            return df
        else:
            raise ValueError("'columns' should be a dict")