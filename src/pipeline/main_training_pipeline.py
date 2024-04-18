"""Module to create Main training pipeline"""
import sys
from src.text_summarizer.pipeline.text_summarizer_training_pipeline\
      import TextSummarizerTrainingPipeline
from src import logger


class MainTrainingPipeline:
    """Class to create Main training pipeline"""

    def __init__(self):
        pass

    def run_pipeline(self, steps: int):
        """Method to perform main training"""
        try:
            # Text Summarizer
            pipeline = TextSummarizerTrainingPipeline()
            pipeline.run_pipeline(steps=steps)
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            NUM_STEPS = int(sys.argv[1])
        else:
            NUM_STEPS = 0
        obj = MainTrainingPipeline()
        obj.run_pipeline(steps=NUM_STEPS)
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
