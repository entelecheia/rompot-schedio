from hyfi.utils.logging import LOGGING
from hyfi.task.batch import BatchTaskConfig
from schedio.tasks.forger.config import ForgerStylize
from typing import Optional

logger = LOGGING.getLogger(__name__)


class ForgerStylizeTask(BatchTaskConfig):
    config_name = "forger_stylize"

    forger: Optional[ForgerStylize] = None

    def paint(self):
        if self.forger is None:
            raise ValueError("No forger config provided.")
        if self.batch is None:
            raise ValueError("No batch config provided.")
        if self.path is None:
            raise ValueError("No path config provided.")

        self.forger.input_dir = str(self.path.input_dir)
        self.forger.model_dir = str(self.path.model_dir)
        self.forger.output_dir = str(self.batch.batch_dir)
        self.forger.tmp_dir = str(self.path.tmp_dir)
        self.forger.output_file_prefix = self.batch.file_prefix
        logger.info("Painting...")
        self.forger.paint()
