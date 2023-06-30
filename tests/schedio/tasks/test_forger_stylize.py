from schedio.tasks.forger import ForgerStylizeTask
from schedio import HyFI


def test_forger_stylize():
    HyFI.initialize()
    forger = ForgerStylizeTask(_config_name_="forger_stylize")
    HyFI.print(forger.dict())
    forger.batch.verbose = True
    forger.paint()


if __name__ == "__main__":
    test_forger_stylize()
