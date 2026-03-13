from comet import download_model, load_from_checkpoint
from typing import List, Dict

class CometEvaluator:
    """Wrapper for COMET model evaluation."""
    
    def __init__(self, model_name: str = "wmt22-comet-da", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Download and load COMET model."""
        model_path = download_model(self.model_name, saving_directory="comet_models")
        self.model = load_from_checkpoint(model_path)
    
    def score(self, src_lines: List[str], mt_lines: List[str], ref_lines: List[str]) -> Dict:
        """
        Score translations using COMET.
        
        Args:
            src_lines: Source language sentences
            mt_lines: Machine-translated sentences
            ref_lines: Reference translations
        
        Returns:
            Dictionary with 'scores' and 'mean_score'
        """
        if not (len(src_lines) == len(mt_lines) == len(ref_lines)):
            raise ValueError("All input lists must have the same length")
        
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(src_lines, mt_lines, ref_lines)
        ]
        
        scores = self.model.predict(data, batch_size=8, gpus=1)
        
        return {
            "scores": scores.scores,
            "mean_score": scores.mean_score,
            "metadata": scores.metadata if hasattr(scores, 'metadata') else None
        }