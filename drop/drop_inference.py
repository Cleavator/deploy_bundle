import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, Any, List, Optional, Tuple, Union
import joblib
import os
from scipy import stats

# Global counter for requests
REQUEST_COUNTER = 0

# Import feature extractor from the sibling module
# Assuming droplet_analyzer_cellpose_v3 is in the same directory
try:
    from .droplet_analyzer_cellpose_v3 import extract_droplet_features_from_array, compute_extended_features_v2, compute_features_mir21, infer_target_from_image, _CP_MODEL_CACHE
except ImportError:
    # Fallback for direct execution or different path structure
    from droplet_analyzer_cellpose_v3 import extract_droplet_features_from_array, compute_extended_features_v2, compute_features_mir21, infer_target_from_image, _CP_MODEL_CACHE

class DropletPredictor:
    """
    Handles droplet analysis, feature engineering, and model inference 
    for miR-21 and miR-92a targets.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir if base_dir else Path(__file__).resolve().parent
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        """Loads SVM models from .pkl files."""
        # Define paths to model files
        # Users should place their .pkl files in the same directory as this script
        paths = {
            "miR-21": self.base_dir / "miR-21-model.joblib",
            "miR-92a": self.base_dir / "model_92a.joblib"
        }
        
        for target, path in paths.items():
            if path.exists():
                try:
                    self.models[target] = joblib.load(path)
                    print(f"[INFO] Loaded SVM model for {target} from {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {target} SVM model: {e}")
            else:
                print(f"[WARNING] Model file not found for {target} at {path}")

    def predict_concentration(self, gray_value: float, target: str) -> Dict[str, float]:
        """
        Calculate concentration using linear regression formula based on Gray Value (y).
        Formula: y = ax + b, where x = -log10(C).
        Thus: x = (y - b) / a
              C = 10 ** (-x)
        
        Returns dict with:
          - log10_concentration
          - concentration_M
          - concentration_fM
          - calib_x (the x value in y=ax+b, which is -log10(C))
          - slope
          - intercept
        """
        if target == "miR-92a":
            # Coefficients for miR-92a
            # y = -2.11009 * x + 44.32428
            a = -2.11009
            b = 44.32428
        elif target == "miR-21":
            # Coefficients for miR-21
            # y = -2.57164 * x + 43.32702
            a = -2.57164
            b = 43.32702
        else:
            return {
                "log10_concentration": 0.0,
                "concentration_M": 0.0,
                "concentration_fM": 0.0,
                "calib_x": 0.0,
                "slope": 0.0,
                "intercept": 0.0
            }
            
        # x = -log10(C)
        # y = ax + b => x = (y - b) / a
        if a != 0:
            x = (gray_value - b) / a
        else:
            x = 0.0
        
        # log10(C) = -x
        log10_conc = -x
        conc_M = 10 ** log10_conc
        conc_fM = conc_M * 1e15
        
        return {
            "log10_concentration": float(log10_conc),
            "concentration_M": float(conc_M),
            "concentration_fM": float(conc_fM),
            "calib_x": float(x),
            "slope": a,
            "intercept": b
        }

    def _extract_features_mir21(self, df: pd.DataFrame) -> np.ndarray:
        """
        Feature extraction for miR-21.
        The complex calculation logic is encapsulated in compute_features_mir21.
        """
        return compute_features_mir21(df)

    def _extract_features_mir92a(self, df: pd.DataFrame) -> np.ndarray:
        """
        Feature extraction for miR-92a.
        The model expects a 46-dimensional feature vector, which includes
        base stats, extended stats, and interaction terms.
        
        The complex calculation logic is encapsulated in compute_extended_features_v2
        to keep the inference code clean.
        """
        return compute_extended_features_v2(df)

    def predict_classification(self, df: pd.DataFrame, target: str) -> Tuple[str, float, Any]:
        """
        Run SVM inference.
        Returns: label, confidence, raw_prediction
        """
        model_container = self.models.get(target)
        
        if model_container is None:
            return "Model not found", 0.0, None
            
        try:
            # Extract features based on target
            if target == "miR-21":
                X = self._extract_features_mir21(df)
            elif target == "miR-92a":
                X = self._extract_features_mir92a(df)
            else:
                return "Unknown Target", 0.0, None
            
            # Handle dict-based pipeline (miR-92a) vs simple model (miR-21)
            if isinstance(model_container, dict):
                # Apply preprocessing steps manually if they exist in the dict
                # Steps: variance_filter -> lasso_selector -> f_mask -> pca -> robust -> minmax -> model
                
                # 1. Variance Threshold
                if "variance_filter" in model_container:
                    X = model_container["variance_filter"].transform(X)
                    
                # 2. Lasso Selection
                if "lasso_selector" in model_container:
                    X = model_container["lasso_selector"].transform(X)
                    
                # 3. F-Mask (Manual Feature Selection mask)
                if "f_mask" in model_container:
                    f_mask = model_container["f_mask"]
                    # If f_mask is indices
                    if hasattr(f_mask, "dtype") and f_mask.dtype == bool:
                         X = X[:, f_mask]
                    else:
                         X = X[:, f_mask]

                # 4. PCA
                if "pca" in model_container:
                    X = model_container["pca"].transform(X)
                    
                # 5. Robust Scaler
                if "robust_scaler" in model_container:
                    X = model_container["robust_scaler"].transform(X)
                    
                # 6. MinMax Scaler
                if "minmax_scaler" in model_container:
                    X = model_container["minmax_scaler"].transform(X)
                
                # Finally, the model
                model = model_container.get("model")
            else:
                # Direct model (Pipeline or Estimator)
                model = model_container

            if model is None:
                 return "Model object missing", 0.0, None

            # Predict
            raw_pred = model.predict(X)[0]
            
            # Get confidence if available
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)[0]
                    confidence = float(np.max(probs))
                except:
                    confidence = 1.0 # Fallback
            elif hasattr(model, "decision_function"):
                dist = model.decision_function(X)[0]
                confidence = float(abs(dist)) 
                
            # Map prediction to label (0=Healthy, 1=Cancer)
            label_map = {0: "Healthy", 1: "Cancer"}
            if isinstance(model_container, dict) and "label_map" in model_container:
                 provided_map = model_container["label_map"]
                 if isinstance(provided_map, dict):
                     first_val = list(provided_map.values())[0]
                     if isinstance(first_val, int):
                         label_map = {v: k for k, v in provided_map.items()}
            
            label = label_map.get(int(raw_pred), f"Class {raw_pred}")
            
            return label, confidence, raw_pred
            
        except Exception as e:
            print(f"[ERROR] SVM Inference failed for {target}: {e}")
            import traceback
            traceback.print_exc()
            return "Inference Error", 0.0, None

    def analyze(self, img: np.ndarray, target: str, ambiguous_policy: str = "error", max_side: int = 768) -> Dict[str, Any]:
        """
        Main entry point for image analysis.
        
        Pipeline:
        1. Cellpose Segmentation -> Get droplet features
        2. Concentration Prediction -> Linear Regression
        3. Disease Classification -> SVM Inference
        """
        import time
        import sys
        import torch
        
        global REQUEST_COUNTER
        REQUEST_COUNTER += 1
        req_id = REQUEST_COUNTER
        
        t_req_start = time.time()
        print(f"\n[REQ] id={req_id} start_time={t_req_start}")
        print(f"\n[REQ] pid={os.getpid()}")
        print(f"[REQ] python={sys.executable}")
        print(f"[REQ] target={target}, ambiguous_policy={ambiguous_policy}")
        print(f"[REQ] max_side={max_side}")
        print(f"[REQ] input_image: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[REQ] device={device}")
        
        if _CP_MODEL_CACHE:
            print(f"[REQ] cp_model cache HIT: keys={list(_CP_MODEL_CACHE.keys())}")
        else:
            print(f"[REQ] cp_model cache MISS (will initialize)")

        print(f"[INFO] Starting analysis for target: {target}, policy: {ambiguous_policy}")
        
        try:
            import cellpose
            cp_ver = getattr(cellpose, "__version__", "unknown")
        except Exception:
            cp_ver = "unknown"
            
        print(f"[INFO] Python: {sys.executable}")
        print(f"[INFO] torch: {torch.__version__}, cellpose: {cp_ver}, mps_available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
        print(f"[INFO] Using device: {device}")
        
        t_all_start = time.time()
        
        # Audit variables
        target_source = "user"
        target_detected = target if target != "auto" else None
        auto_score = None
        auto_R_top = None
        auto_G_top = None
        ambiguous_reason = None
        
        # Auto-detect target if requested
        if target == "auto":
            t_auto_start = time.time()
            print("[INFO] Auto-detecting target...")
            detected_target, score, r_top, g_top = infer_target_from_image(img)
            t_auto_end = time.time()
            print(f"[INFO] Auto-detect took {t_auto_end - t_auto_start:.3f}s")
            
            auto_score = score
            auto_R_top = r_top
            auto_G_top = g_top
            target_source = "auto"
            
            if detected_target == "ambiguous":
                ambiguous_reason = f"score={score:.3f}" if score is not None else "format error"
                print(f"[WARNING] Ambiguous target ({ambiguous_reason}). Policy={ambiguous_policy}")
                
                if ambiguous_policy == "error":
                    err_msg = f"Auto-target ambiguous; please select miR-21 or miR-92a. (Reason: {ambiguous_reason})"
                    print(f"[ERROR] {err_msg}")
                    raise ValueError(err_msg)
                elif ambiguous_policy == "skip":
                    return {
                        "target": "ambiguous",
                        "raw_prediction": "Skipped",
                        "label": "Skipped (Ambiguous)",
                        "confidence": 0.0,
                        "n_droplets": 0,
                        "log10_concentration": None,
                        "concentration_M": None,
                        "concentration_fM": None,
                        "concentration": None,
                        "target_detected": "ambiguous",
                        "target_source": "auto",
                        "auto_target_score": score,
                        "ambiguous_reason": ambiguous_reason
                    }
                elif ambiguous_policy == "default21":
                    target = "miR-21"
                    target_source = "auto_default"
                    print("[INFO] Using default: miR-21")
                elif ambiguous_policy == "default92a":
                    target = "miR-92a"
                    target_source = "auto_default"
                    print("[INFO] Using default: miR-92a")
            else:
                target = detected_target
                target_detected = target
                print(f"[INFO] Auto-detected target: {target} (score={score:.3f})")
        
        # Step 1: Segmentation and Feature Extraction
        t_seg_start = time.time()
        droplet_df = extract_droplet_features_from_array(img, model_name="cyto3", max_side=int(max_side))
        t_seg_end = time.time()
        print(f"[INFO] Segmentation+features took {t_seg_end - t_seg_start:.3f}s")
        n_droplets = len(droplet_df)
        print(f"[INFO] Detected {n_droplets} droplets.")
        
        if n_droplets == 0:
            return {
                "target": target,
                "raw_prediction": "Unknown",
                "label": "Unknown (No droplets)",
                "confidence": 0.0,
                "n_droplets": 0,
                "log10_concentration": None,
                "concentration_M": None,
                "concentration_fM": None,
                "concentration": None,
                "target_detected": target_detected or target,
                "target_source": target_source,
                "auto_target_score": auto_score,
                "calib_slope_used": None,
                "calib_intercept_used": None
            }
            
        # Step 2: Concentration Prediction
        current_gray_value = droplet_df['gray_mean'].mean()
        t_conc_start = time.time()
        conc_results = self.predict_concentration(current_gray_value, target)
        t_conc_end = time.time()
        print(f"[INFO] Concentration calc took {t_conc_end - t_conc_start:.3f}s")
        
        log10_conc = conc_results["log10_concentration"]
        conc_M = conc_results["concentration_M"]
        conc_fM = conc_results["concentration_fM"]
        calib_x = conc_results["calib_x"]
        
        print(f"[INFO] Gray Value: {current_gray_value:.2f} -> Log10 Conc: {log10_conc:.4f}")
        
        # Step 3: Disease Classification
        t_pred_start = time.time()
        label, confidence, raw_pred = self.predict_classification(droplet_df, target)
        t_pred_end = time.time()
        print(f"[INFO] Prediction took {t_pred_end - t_pred_start:.3f}s")
        print(f"[INFO] Total analyze took {t_pred_end - t_all_start:.3f}s")
        print(f"[INFO] Prediction: {label} (raw={raw_pred}), Conf: {confidence:.2f}")
        
        t_total_req = time.time() - t_req_start
        print(f"[REQ_DONE] id={req_id} total={t_total_req:.4f}s")
        
        # Add calibration details to return
        conc_results = self.predict_concentration(current_gray_value, target_detected or target)
        
        return {
            "target": target,
            "raw_prediction": raw_pred,
            "label": label,
            "confidence": confidence,
            "n_droplets": n_droplets,
            "log10_concentration": log10_conc,
            "concentration_M": conc_M,
            "concentration_fM": conc_fM,
            "concentration": log10_conc,
            "calib_x": calib_x,
            "gray_value_used_for_calibration": float(current_gray_value),
            "calib_slope_used": conc_results["slope"],
            "calib_intercept_used": conc_results["intercept"],
            "target_detected": target_detected or target,
            "target_source": target_source,
            "auto_target_score": auto_score,
            "auto_R_top": auto_R_top,
            "auto_G_top": auto_G_top,
            "ambiguous_reason": ambiguous_reason
        }

# Singleton instance for easy import
predictor = DropletPredictor()

# Expose the analyze method as the main function
full_droplet_analysis = predictor.analyze
