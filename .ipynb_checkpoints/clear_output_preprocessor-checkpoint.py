import nbformat
from nbconvert.preprocessors import Preprocessor

class ClearOutputPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        for cell in nb.cells:
            if "outputs" in cell:
                cell["outputs"] = []
        return nb, resources

