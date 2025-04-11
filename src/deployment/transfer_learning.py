# src/deployment/transfer_learning.py
import torch
import numpy as np
import logging
import traceback
import os
import json
from stable_baselines3 import PPO
from typing import Optional

# Robot config constants
from config import robot_config

logger = logging.getLogger(__name__)

# Epsilon for division by zero
EPSILON = 1e-8

# Default parameters file
DEFAULT_PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'transfer_params.json')

# Default velocity limits (rad/s)
DEFAULT_VELOCITY_LIMITS_RAD_S = robot_config.VELOCITY_LIMITS_RAD_S

class RobotTransfer:
    """Adapts trained model I/O for real robot deployment."""
    def __init__(self, model_path: str,
                 state_mean: np.ndarray | None = None,
                 state_std: np.ndarray | None = None,
                 action_scale: np.ndarray | None = None,
                 action_offset: np.ndarray | None = None,
                 params_file: str = DEFAULT_PARAMS_FILE,
                 velocity_limits: Optional[np.ndarray] = None):
        """
        Args:
            model_path: Path to model (.zip).
            state_mean: State normalisation mean.
            state_std: State normalisation std dev.
            action_scale: Action scaling factor.
            action_offset: Action offset.
            params_file: Path to JSON calibration params.
            velocity_limits: Max joint velocities (rad/s).
        """
        # Path to the trained model
        self.model_path: str = model_path
        # Loaded RL model instance
        self.model: PPO | None = None

        # Observation and action dimensions
        self.obs_dim: int = 21
        self.action_dim: int = 5

        # State normalisation parameters
        self.state_mean = np.array(state_mean, dtype=np.float32) if state_mean is not None else None
        self.state_std = np.array(state_std, dtype=np.float32) if state_std is not None else None
        # Action correction parameters
        self.action_scale = np.array(action_scale, dtype=np.float32) if action_scale is not None else None
        self.action_offset = np.array(action_offset, dtype=np.float32) if action_offset is not None else None
        # Calibration parameters file path
        self.params_file = params_file

        # Joint velocity limits (rad/s)
        self.velocity_limits = np.array(velocity_limits, dtype=np.float32) if velocity_limits is not None else DEFAULT_VELOCITY_LIMITS_RAD_S
        if velocity_limits is None:
             logger.warning("Velocity limits not provided to RobotTransfer, using default values for clipping.")
        elif len(self.velocity_limits) != self.action_dim:
             logger.error(f"Provided velocity_limits dimension ({len(self.velocity_limits)}) != action_dim ({self.action_dim}). Using defaults.")
             self.velocity_limits = DEFAULT_VELOCITY_LIMITS_RAD_S

        # Calibration status
        self.is_calibrated: bool = False

        # Load model and calibration parameters
        self.model = self._load_model()

        if self.load_calibration_params(self.params_file):
             self.is_calibrated = True

    def _load_model(self):
        """Loads RL model."""
        try:
            model = PPO.load(self.model_path, device='cpu')
            logger.info(f"Successfully loaded trained model from: {self.model_path}")
            return model
        except FileNotFoundError:
             logger.error(f"Model file not found at {self.model_path}. Ensure path is correct.")
             return None
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    def load_calibration_params(self, filepath: str) -> bool:
        """
        Loads calibration params from JSON.

        Args:
            filepath: Path to JSON.

        Returns:
            bool: True if loaded.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Calibration parameter file not found at: {filepath}. Using defaults or previously set values.")
            return False

        try:
            with open(filepath, 'r') as f:
                params = json.load(f)

            # Flag for successful parameter loading
            loaded_something = False
            # Temporary storage for loaded parameters
            temp_state_mean = self.state_mean
            temp_state_std = self.state_std
            temp_action_scale = self.action_scale
            temp_action_offset = self.action_offset

            if 'state_mean' in params and params['state_mean'] is not None:
                temp_state_mean = np.array(params['state_mean'], dtype=np.float32)
                logger.info(f"Loaded state_mean from {filepath}")
                loaded_something = True
            if 'state_std' in params and params['state_std'] is not None:
                temp_state_std = np.array(params['state_std'], dtype=np.float32)
                if temp_state_std is not None:
                     temp_state_std[temp_state_std == 0] = EPSILON
                logger.info(f"Loaded state_std from {filepath}")
                loaded_something = True
            if 'action_scale' in params and params['action_scale'] is not None:
                temp_action_scale = np.array(params['action_scale'], dtype=np.float32)
                logger.info(f"Loaded action_scale from {filepath}")
                loaded_something = True
            if 'action_offset' in params and params['action_offset'] is not None:
                temp_action_offset = np.array(params['action_offset'], dtype=np.float32)
                logger.info(f"Loaded action_offset from {filepath}")
                loaded_something = True

            if not loaded_something:
                 logger.warning(f"Calibration file {filepath} loaded, but contained no valid parameter keys (state_mean, state_std, action_scale, action_offset).")
                 return False

            if self.model:
                assert self.model.observation_space.shape is not None, "Model observation space shape is None after load"
                obs_dim = self.model.observation_space.shape[0]
                assert self.model.action_space.shape is not None, "Model action space shape is None after load"
                act_dim = self.model.action_space.shape[0]

                # Validation flag for parameter dimensions
                valid = True
                if temp_state_mean is not None and len(temp_state_mean) != obs_dim:
                     logger.error(f"Loaded state_mean dimension ({len(temp_state_mean)}) mismatch model observation dim ({obs_dim}). Check {filepath}.")
                     valid = False
                if temp_state_std is not None and len(temp_state_std) != obs_dim:
                     logger.error(f"Loaded state_std dimension ({len(temp_state_std)}) mismatch model observation dim ({obs_dim}). Check {filepath}.")
                     valid = False
                if temp_action_scale is not None and len(temp_action_scale) != act_dim:
                     logger.error(f"Loaded action_scale dimension ({len(temp_action_scale)}) mismatch model action dim ({act_dim}). Check {filepath}.")
                     valid = False
                if temp_action_offset is not None and len(temp_action_offset) != act_dim:
                     logger.error(f"Loaded action_offset dimension ({len(temp_action_offset)}) mismatch model action dim ({act_dim}). Check {filepath}.")
                     valid = False

                if not valid:
                     logger.error("Calibration parameter validation failed. Parameters NOT loaded.")
                     return False
            else:
                logger.warning("Model not loaded, cannot validate parameter dimensions.")

            # Update instance parameters after validation
            self.state_mean = temp_state_mean
            self.state_std = temp_state_std
            self.action_scale = temp_action_scale
            self.action_offset = temp_action_offset
            self.is_calibrated = True

            logger.info(f"Successfully loaded and validated calibration parameters from {filepath}.")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from calibration file {filepath}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading calibration parameters from {filepath}: {e}")
            return False

    def calibrate(self, sim_states: np.ndarray | None = None, real_states: np.ndarray | None = None,
                  sim_actions: np.ndarray | None = None, real_actions: np.ndarray | None = None):
        """
        Calibrates using paired sim/real data.

        Args:
            sim_states: Sim states (N x obs_dim).
            real_states: Real states (N x obs_dim).
            sim_actions: Sim actions (N x action_dim).
            real_actions: Real actions (N x action_dim).
        """
        logger.info("Attempting calibration with provided data...")
        # Flag for calibration completion
        calibration_performed = False

        # Update state normalisation with real data
        if real_states is not None and len(real_states) > 1:
            if real_states.shape[1] != self.obs_dim:
                logger.error(f"Calibration Error: real_states dimension mismatch ({real_states.shape[1]} != expected {self.obs_dim}). Skipping state normalization update.")
            else:
                new_mean = np.mean(real_states, axis=0, dtype=np.float32)
                new_std = np.std(real_states, axis=0, dtype=np.float32)
                new_std[new_std < EPSILON] = 1.0
                self.state_mean = new_mean
                self.state_std = new_std
                logger.info(f"Updated state normalization (mean/std) based on {len(real_states)} real samples.")
                calibration_performed = True
        else:
            logger.warning(f"Insufficient real_states data provided ({len(real_states) if real_states is not None else 'None'} samples). State normalization parameters remain unchanged.")

        # Initialise action correction parameters if not set
        if self.action_scale is None:
            self.action_scale = np.ones(self.action_dim, dtype=np.float32)
        if self.action_offset is None:
            self.action_offset = np.zeros(self.action_dim, dtype=np.float32)

        # Calculate action correction with paired data
        if (sim_actions is not None and real_actions is not None and
            len(sim_actions) > 1 and len(real_actions) == len(sim_actions)):

            if sim_actions.shape[1] != self.action_dim or real_actions.shape[1] != self.action_dim:
                 logger.error(f"Calibration Error: action dimensions mismatch (sim:{sim_actions.shape[1]}, real:{real_actions.shape[1]} vs expected {self.action_dim}). Skipping action correction update.")
            else:
                logger.info(f"Calculating action correction (scale/offset) based on {len(sim_actions)} paired samples.")
                # Count of corrected action dimensions
                corrected_count = 0
                for i in range(self.action_dim):
                    sim_a = sim_actions[:, i]
                    real_a = real_actions[:, i]

                    if np.std(sim_a) < EPSILON:
                        logger.warning(f"Skipping action correction for action dim {i}: sim action variance is near zero. Using default scale=1, offset=0.")
                        self.action_scale[i] = 1.0
                        self.action_offset[i] = 0.0
                        continue

                    try:
                        A = np.vstack([sim_a, np.ones(len(sim_a))]).T
                        scale, offset = np.linalg.lstsq(A, real_a, rcond=None)[0]

                        if abs(scale) > 5.0 or abs(scale) < 0.2:
                             logger.warning(f"Suspicious action scale ({scale:.3f}) calculated for dim {i}. Check calibration data or consider non-linear correction.")
                        if abs(offset) > 1.0:
                             logger.warning(f"Large action offset ({offset:.3f}) calculated for dim {i}. Check calibration data or units.")

                        self.action_scale[i] = float(scale)
                        self.action_offset[i] = float(offset)
                        corrected_count += 1
                    except np.linalg.LinAlgError as e:
                        logger.error(f"Linear regression failed for action dim {i}: {e}. Using default scale=1, offset=0.")
                        self.action_scale[i] = 1.0
                        self.action_offset[i] = 0.0
                    except Exception as e:
                         logger.error(f"Unexpected error during action dim {i} fit: {e}. Using defaults.")
                         self.action_scale[i] = 1.0
                         self.action_offset[i] = 0.0

                if corrected_count > 0:
                    logger.info(f"Action correction parameters (scale/offset) updated for {corrected_count}/{self.action_dim} dimensions.")
                    calibration_performed = True
        else:
            logger.warning("Insufficient or mismatched sim/real action data provided. Action correction parameters remain unchanged.")

        self.is_calibrated = calibration_performed
        if self.is_calibrated:
             logger.info("Calibration completed with available data.")
        else:
             logger.warning("Calibration could not be performed with the provided data. Using initial default parameters.")

    def normalize_state(self, state: np.ndarray) -> np.ndarray | None:
        """Normalises real state."""
        if state is None or state.shape != (self.obs_dim,):
            logger.error(f"Invalid state shape for normalization: expected ({self.obs_dim},), got {state.shape if state is not None else 'None'}.")
            return None

        # Initialise normalisation parameters if not set
        if self.state_mean is None:
            self.state_mean = np.zeros(self.obs_dim, dtype=np.float32)
            logger.warning("State mean was None in normalize_state, initializing to zeros.")
        if self.state_std is None:
            self.state_std = np.ones(self.obs_dim, dtype=np.float32)
            logger.warning("State std was None in normalize_state, initializing to ones.")

        try:
            if self.state_std is None or self.state_mean is None:
                 logger.error("State mean or std is unexpectedly None during normalization.")
                 return None
            # Apply normalisation formula
            normalized_state = (state - self.state_mean) / self.state_std
            return normalized_state.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during state normalization: {e}")
            logger.error(traceback.format_exc())
            return None

    def correct_action(self, action: np.ndarray) -> np.ndarray | None:
        """Corrects sim action."""
        if action is None or action.shape != (self.action_dim,):
            logger.error(f"Invalid action shape for correction: expected ({self.action_dim},), got {action.shape if action is not None else 'None'}.")
            return None

        # Initialise correction parameters if not set
        if self.action_scale is None:
            self.action_scale = np.ones(self.action_dim, dtype=np.float32)
            logger.warning("Action scale was None in correct_action, initializing to ones.")
        if self.action_offset is None:
            self.action_offset = np.zeros(self.action_dim, dtype=np.float32)
            logger.warning("Action offset was None in correct_action, initializing to zeros.")

        try:
            # Apply correction formula
            corrected_action = action * self.action_scale + self.action_offset

            # Clip actions to velocity limits
            if self.velocity_limits is not None:
                 corrected_action = np.clip(corrected_action, -self.velocity_limits, self.velocity_limits)
            else:
                 logger.warning("Velocity limits not available for action clipping.")

            return corrected_action.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during action correction: {e}")
            logger.error(traceback.format_exc())
            return None

    def predict(self, real_state: np.ndarray) -> np.ndarray | None:
        """
        Predicts real action from real state.

        Args:
            real_state: Real state (obs_dim).

        Returns:
            np.ndarray | None: Corrected action (action_dim).
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return None
        if not self.is_calibrated:
             logger.warning("Predicting using uncalibrated transfer parameters. Results may be inaccurate.")

        # Normalise input state
        normalized_state = self.normalize_state(real_state)
        if normalized_state is None:
            logger.error("State normalization failed. Cannot predict.")
            return None

        try:
            # Predict simulated action
            action_sim, _ = self.model.predict(normalized_state, deterministic=True)
        except Exception as e:
            logger.error(f"Error during model.predict(): {e}")
            logger.error(traceback.format_exc())
            return None

        # Correct simulated action for real robot
        corrected_action = self.correct_action(action_sim)
        if corrected_action is None:
            logger.error("Action correction failed. Cannot provide final action.")
            return None

        return corrected_action