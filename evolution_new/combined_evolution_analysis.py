from analysis_pipeline import AnalysisPipeline
from analysis_config import *

if __name__ == "__main__":
    analysis_pipeline = AnalysisPipeline(analysis_pipeline_sgi_big)
    analysis_pipeline.start_visualisation()
