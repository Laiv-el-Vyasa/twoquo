from analysis_pipeline import AnalysisPipeline
from analysis_config import *

if __name__ == "__main__":
    analysis_pipeline = AnalysisPipeline(scale_analysis_pipeline)
    analysis_pipeline.start_visualisation()
