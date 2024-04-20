"""Module to create Main prediction pipeline"""
from pathlib import Path
import pandas as pd
from src.car_detection.pipeline.car_detection_prediction_pipeline import predict_cd
from src.damage_detection.pipeline.damage_detection_prediction_pipeline import predict_dd
from src.damage_severity.pipeline.damage_severity_prediction_pipeline import predict_ds


class MainPredictionPipeline:
    """Class to create Main prediction pipeline"""

    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def get_repair_price(damage_result, severity_result):
        """Method to get the repair price"""
        try:
            repair_cost_df = pd.read_csv(Path("Data/repair_cost_data.csv"))
            repair_cost_df = repair_cost_df.loc[\
                repair_cost_df['damage_type'] == damage_result]
            repair_cost_df = repair_cost_df.loc[\
                repair_cost_df['damage_sev'] == severity_result]
            repair_price = repair_cost_df['repair_cost'].values[0]
            repair_price = "The average estimated repair cost is: $" + str(repair_price)
            return repair_price
        except Exception as ex:
            raise ex

    def run_pipeline(self):
        """Method to perform prediction"""
        try:
            car_result = [False, ""]
            damage_result = [False, ""]
            severity_result = ""
            repair_price = ""
            # Car Detection
            car_result = predict_cd(self.filename)
            if car_result[0] is True:
                # Damage Detection
                damage_result = predict_dd(self.filename)
                if damage_result[0] is True:
                    # Severity Detection
                    severity_result = predict_ds(self.filename)
                    repair_price = self.get_repair_price(damage_result[1].split(":")[1].strip(),
                                                         severity_result.split(":")[1].strip())
            return car_result[1], damage_result[1], severity_result, repair_price
        except Exception as ex:
            raise ex
