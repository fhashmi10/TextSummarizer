"""
main module
"""
import sys
from src import logger
from src.pipeline.main_training_pipeline import MainTrainingPipeline

try:
    obj = MainTrainingPipeline()
    if len(sys.argv)>1:
        NUM_STEPS = int(sys.argv[1])
    else:
        NUM_STEPS = 0
    obj.run_pipeline(steps=NUM_STEPS)
except Exception as ex:
    logger.exception("Exception in processing: %s", ex)
