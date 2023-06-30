from hyfi.utils.logging import LOGGING
from hyfi.task.batch import BatchTaskConfig
from schedio.tasks.forger.config import ForgerStylize
from typing import Optional

logger = LOGGING.getLogger(__name__)


class ForgerStylizeTask(BatchTaskConfig):
    _config_name_ = "forger_stylize"

    forger: Optional[ForgerStylize] = None

    def initialize_configs(self, **config_kwargs):
        super().initialize_configs(**config_kwargs)
        print(config_kwargs)
        if "forger" in self.__dict__ and self.__dict__["forger"]:
            self.forger = ForgerStylize.parse_obj(self.__dict__["forger"])

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
