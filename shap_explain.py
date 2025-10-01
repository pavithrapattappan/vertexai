from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

class shap_explain():
    def __init__(self, session, data, x_cols, y_col, runid, runtimestamp):
        self.session = session
        self.data = data
        self.x_cols = x_cols
        self.y_col = y_col
        self.runid = runid
        self.runtimestamp = runtimestamp
        
    def shap_explain(self):
        data_pd = self.data.to_pandas().dropna()
        results=[]
        for s in data_pd["SEGMENT"].unique():
            segment_df = data_pd[data_pd["SEGMENT"] == s]
            X = segment_df[self.x_cols]  
            y = segment_df[self.y_col]

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X, y)

            # Compute SHAP values
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X)

            for c in list(range(int(y["CLUSTER_LABEL"].max()+1))):
                cluster_shap = shap_values.values[:, :, c]
                plt.figure()
                shap.summary_plot(cluster_shap, X, show=False)
                buf= io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                html_plot = f'<img src="data:image/png;base64,{img_str}"/>'

                results.append({
                    'SEGMENT': s,
                    'CLUSTER': c,
                    'CLUSTER_SHAP': cluster_shap.tolist(),  # Convert to list for DataFrame
                    'CLUSTER_SHAP_PLOT': html_plot
                })
        self.results_df = pd.DataFrame(results)
        
    def add_stamps(self):
        self.results_df["RUNID"] = self.runid
        self.results_df["RUNTIMESTAMP"] = self.runtimestamp
    
    def create_snowpark_df(self):
        self.snowpark_df = self.session.create_dataframe(self.results_df)
        
    def save_shap_exp(self, table):
        self.snowpark_df.write.mode("append").save_as_table(table)
        
    def run_shap_explain(self, table):
        self.shap_explain()
        self.add_stamps()
        self.create_snowpark_df()
        self.save_shap_exp(table)