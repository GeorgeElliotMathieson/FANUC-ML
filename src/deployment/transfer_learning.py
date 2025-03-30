# src/transfer_learning.py
import torch
import numpy as np
import logging
import traceback # Added for better error logging
import os
import json # Import json for loading parameters
from stable_baselines3 import PPO
from typing import Optional # Import Optional for type hinting

# Import robot config constants
from config import robot_config

logger = logging.getLogger(__name__)

# Epsilon value to prevent division by zero in normalization/calibration
EPSILON = 1e-8

# --- Constants (Example values - Adjust as needed) ---
DEFAULT_STATE_MEAN: Optional[np.ndarray] = None # Defaults to no normalization if None
DEFAULT_STATE_STD: Optional[np.ndarray] = None  # Defaults to no normalization if None
DEFAULT_ACTION_SCALE: Optional[np.ndarray] = None # Defaults to no scaling if None
DEFAULT_ACTION_OFFSET: Optional[np.ndarray] = None # Defaults to no offset if None
# Define path relative to project root (two levels up from src/deployment/)
DEFAULT_PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'transfer_params.json')
# Use default velocity limits from config
DEFAULT_VELOCITY_LIMITS_RAD_S = robot_config.VELOCITY_LIMITS_RAD_S

class RobotTransfer:
    """
    Handles loading a trained SB3 model and adapting its inputs/outputs
    for deployment on a real robot. Includes basic state normalization and
    action correction placeholders.

    **CRITICAL WARNINGS:**
    1.  **Calibration Method:** The calibration methods provided (mean/std state
        normalization, linear action correction via least squares) are VERY basic
        placeholders. Real-world sim-to-real often requires more sophisticated
        techniques like domain randomization during training, learning residual
        dynamics models, careful system identification, or adaptive methods.
    2.  **Calibration Data Quality:** The effectiveness of ANY calibration heavily
        depends on the quality and coverage of the collected paired simulation
        and real-world data. Data must be accurate, time-aligned, and cover
        the expected operating range.
    3.  **Model Compatibility:** Assumes the loaded model's observation and action
        spaces match the dimensions defined here (`obs_dim`, `action_dim`) and
        the data provided by the `robot_api.py`.
    """
    def __init__(self, model_path: str,
                 state_mean: np.ndarray | None = DEFAULT_STATE_MEAN,
                 state_std: np.ndarray | None = DEFAULT_STATE_STD,
                 action_scale: np.ndarray | None = DEFAULT_ACTION_SCALE,
                 action_offset: np.ndarray | None = DEFAULT_ACTION_OFFSET,
                 params_file: str = DEFAULT_PARAMS_FILE,
                 velocity_limits: Optional[np.ndarray] = None):
        """
        Initialises the transfer module.

        Args:
            model_path: Path to the trained RL model (.zip file).
            state_mean: Optional mean for state normalization.
            state_std: Optional standard deviation for state normalization.
            action_scale: Optional scaling factor for actions.
            action_offset: Optional offset for actions.
            params_file: Path to the JSON file containing calibration parameters.
            velocity_limits: Optional array of maximum joint velocities (rad/s) for clipping.
        """
        self.model_path: str = model_path
        self.model: PPO | None = None # Assuming PPO, specify correct type if different

        # Placeholder dimensions based on src/fanuc_env.py - **VERIFY THESE**
        self.obs_dim: int = 21
        self.action_dim: int = 5 # Assuming 5 controlled joints

        self.state_mean = np.array(state_mean, dtype=np.float32) if state_mean is not None else None
        self.state_std = np.array(state_std, dtype=np.float32) if state_std is not None else None
        self.action_scale = np.array(action_scale, dtype=np.float32) if action_scale is not None else None
        self.action_offset = np.array(action_offset, dtype=np.float32) if action_offset is not None else None
        self.params_file = params_file

        # Store velocity limits for clipping
        self.velocity_limits = np.array(velocity_limits, dtype=np.float32) if velocity_limits is not None else DEFAULT_VELOCITY_LIMITS_RAD_S
        if velocity_limits is None:
             logger.warning("Velocity limits not provided to RobotTransfer, using default values for clipping.")
        elif len(self.velocity_limits) != self.action_dim:
             logger.error(f"Provided velocity_limits dimension ({len(self.velocity_limits)}) != action_dim ({self.action_dim}). Using defaults.")
             self.velocity_limits = DEFAULT_VELOCITY_LIMITS_RAD_S

        self.is_calibrated: bool = False # Flag to track if calibration has been performed

        self.model = self._load_model()

        # Attempt to load calibration params from file on initialization
        if self.load_calibration_params(self.params_file):
             self.is_calibrated = True # Mark as calibrated if params loaded

    def _load_model(self):
        """Load the trained RL model from the specified path."""
        try:
            # Specify device explicitly if needed, e.g., device='cpu' if deployment machine has no GPU
            model = PPO.load(self.model_path, device='cpu')
            logger.info(f"Successfully loaded trained model from: {self.model_path}")
            # Optional: Check model's observation/action space compatibility here?
            # if self.model.observation_space.shape[0] != self.obs_dim or \
            #    self.model.action_space.shape[0] != self.action_dim:
            #     logger.warning("Model's spaces might not match configured dimensions!")
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
        Loads calibration parameters (mean, std, scale, offset) from a JSON file.
        Also sets the is_calibrated flag to True if successful.

        Args:
            filepath: Path to the JSON file.

        Returns:
            bool: True if parameters were loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Calibration parameter file not found at: {filepath}. Using defaults or previously set values.")
            return False

        try:
            with open(filepath, 'r') as f:
                params = json.load(f)

            loaded_something = False
            # Use temporary variables to avoid partially overwriting existing valid params on error
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
                # Add small epsilon to avoid division by zero
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

            # Basic validation (dimensions should match model)
            if self.model:
                # Add assertions for shape existence
                assert self.model.observation_space.shape is not None, "Model observation space shape is None after load"
                obs_dim = self.model.observation_space.shape[0]
                assert self.model.action_space.shape is not None, "Model action space shape is None after load"
                act_dim = self.model.action_space.shape[0]

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
                     return False # Parameters not loaded due to validation error
            else:
                logger.warning("Model not loaded, cannot validate parameter dimensions.")

            # Assign validated parameters
            self.state_mean = temp_state_mean
            self.state_std = temp_state_std
            self.action_scale = temp_action_scale
            self.action_offset = temp_action_offset
            self.is_calibrated = True # Mark as calibrated

            logger.info(f"Successfully loaded and validated calibration parameters from {filepath}.")
            return True # Parameters loaded and validated

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from calibration file {filepath}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading calibration parameters from {filepath}: {e}")
            return False

    def calibrate(self, sim_states: np.ndarray | None = None, real_states: np.ndarray | None = None,
                  sim_actions: np.ndarray | None = None, real_actions: np.ndarray | None = None):
        """
        Calibrate transfer parameters using paired simulation and real-world data.

        **Requires:**
        - `real_states`: Used to calculate mean/std for state normalization.
        - `sim_actions`, `real_actions`: Used for linear fit to find action scale/offset.
          Data points must correspond (i.e., `real_actions[i]` resulted from `sim_actions[i]`
          under state `sim_states[i]`, approximately `real_states[i]`).

        Args:
            sim_states: Array of simulation states (N x obs_dim). Optional.
            real_states: Array of corresponding real robot states (N x obs_dim). Needed for state norm.
            sim_actions: Array of simulation actions (N x action_dim). Needed for action corr.
            real_actions: Array of corresponding real robot actions (N x action_dim). Needed for action corr.
        """
        logger.info("Attempting calibration with provided data...")
        calibration_performed = False

        # --- State Normalization Calibration (Requires Real States) --- 
        if real_states is not None and len(real_states) > 1:
            if real_states.shape[1] != self.obs_dim:
                logger.error(f"Calibration Error: real_states dimension mismatch ({real_states.shape[1]} != expected {self.obs_dim}). Skipping state normalization update.")
            else:
                new_mean = np.mean(real_states, axis=0, dtype=np.float32)
                new_std = np.std(real_states, axis=0, dtype=np.float32)
                # Prevent zero standard deviation
                new_std[new_std < EPSILON] = 1.0
                self.state_mean = new_mean
                self.state_std = new_std
                logger.info(f"Updated state normalization (mean/std) based on {len(real_states)} real samples.")
                # logger.debug(f"  New state_mean: {self.state_mean}")
                # logger.debug(f"  New state_std: {self.state_std}")
                calibration_performed = True
        else:
            logger.warning(f"Insufficient real_states data provided ({len(real_states) if real_states is not None else 'None'} samples). State normalization parameters remain unchanged.")

        # --- Action Correction Calibration (Requires Paired Sim/Real Actions) --- 
        # Ensure scale/offset are initialized as arrays before potential assignment
        if self.action_scale is None:
            self.action_scale = np.ones(self.action_dim, dtype=np.float32)
        if self.action_offset is None:
            self.action_offset = np.zeros(self.action_dim, dtype=np.float32)

        if (sim_actions is not None and real_actions is not None and
            len(sim_actions) > 1 and len(real_actions) == len(sim_actions)): # Need at least 2 points for fit

            if sim_actions.shape[1] != self.action_dim or real_actions.shape[1] != self.action_dim:
                 logger.error(f"Calibration Error: action dimensions mismatch (sim:{sim_actions.shape[1]}, real:{real_actions.shape[1]} vs expected {self.action_dim}). Skipping action correction update.")
            else:
                logger.info(f"Calculating action correction (scale/offset) based on {len(sim_actions)} paired samples.")
                corrected_count = 0
                for i in range(self.action_dim):
                    sim_a = sim_actions[:, i]
                    real_a = real_actions[:, i]

                    # Avoid fitting if sim action variance is near zero
                    if np.std(sim_a) < EPSILON:
                        logger.warning(f"Skipping action correction for action dim {i}: sim action variance is near zero. Using default scale=1, offset=0.")
                        self.action_scale[i] = 1.0
                        self.action_offset[i] = 0.0
                        continue

                    try:
                        # Linear regression: real_a = scale * sim_a + offset
                        A = np.vstack([sim_a, np.ones(len(sim_a))]).T
                        scale, offset = np.linalg.lstsq(A, real_a, rcond=None)[0]

                        # Basic sanity check on scale/offset (adjust thresholds as needed)
                        if abs(scale) > 5.0 or abs(scale) < 0.2:
                             logger.warning(f"Suspicious action scale ({scale:.3f}) calculated for dim {i}. Check calibration data or consider non-linear correction.")
                        if abs(offset) > 1.0: # Large offset might indicate issues (units: radians or rad/s?)
                             logger.warning(f"Large action offset ({offset:.3f}) calculated for dim {i}. Check calibration data or units.")

                        self.action_scale[i] = float(scale)
                        self.action_offset[i] = float(offset)
                        corrected_count += 1
                    except np.linalg.LinAlgError as e:
                        logger.error(f"Linear regression failed for action dim {i}: {e}. Using default scale=1, offset=0.")
                        self.action_scale[i] = 1.0 # Reset to default
                        self.action_offset[i] = 0.0
                    except Exception as e:
                         logger.error(f"Unexpected error during action dim {i} fit: {e}. Using defaults.")
                         self.action_scale[i] = 1.0
                         self.action_offset[i] = 0.0

                if corrected_count > 0:
                    logger.info(f"Action correction parameters (scale/offset) updated for {corrected_count}/{self.action_dim} dimensions.")
                    # logger.debug(f"  New action_scale: {self.action_scale}")
                    # logger.debug(f"  New action_offset: {self.action_offset}")
                    calibration_performed = True
        else:
            logger.warning("Insufficient or mismatched sim/real action data provided. Action correction parameters remain unchanged.")

        self.is_calibrated = calibration_performed
        if self.is_calibrated:
             logger.info("Calibration completed with available data.")
        else:
             logger.warning("Calibration could not be performed with the provided data. Using initial default parameters.")

    def normalize_state(self, state: np.ndarray) -> np.ndarray | None:
        """Normalize a state from the real robot using calibrated mean and std."""
        if state is None or state.shape != (self.obs_dim,):
            logger.error(f"Invalid state shape for normalization: expected ({self.obs_dim},), got {state.shape if state is not None else 'None'}.")
            return None

        # Ensure mean/std are initialized
        if self.state_mean is None:
            self.state_mean = np.zeros(self.obs_dim, dtype=np.float32)
            logger.warning("State mean was None in normalize_state, initializing to zeros.")
        if self.state_std is None:
            self.state_std = np.ones(self.obs_dim, dtype=np.float32)
            logger.warning("State std was None in normalize_state, initializing to ones.")

        try:
            # Use stored mean/std. std already has epsilon added in load/calibrate
            # Check again for safety before division
            if self.state_std is None or self.state_mean is None: # Should not happen after checks above
                 logger.error("State mean or std is unexpectedly None during normalization.")
                 return None
            normalized_state = (state - self.state_mean) / self.state_std
            return normalized_state.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during state normalization: {e}")
            logger.error(traceback.format_exc())
            return None

    def correct_action(self, action: np.ndarray) -> np.ndarray | None:
        """Correct a raw action from the simulation model using calibrated scale and offset."""
        if action is None or action.shape != (self.action_dim,):
            logger.error(f"Invalid action shape for correction: expected ({self.action_dim},), got {action.shape if action is not None else 'None'}.")
            return None

        # Ensure scale/offset are initialized
        if self.action_scale is None:
            self.action_scale = np.ones(self.action_dim, dtype=np.float32)
            logger.warning("Action scale was None in correct_action, initializing to ones.")
        if self.action_offset is None:
            self.action_offset = np.zeros(self.action_dim, dtype=np.float32)
            logger.warning("Action offset was None in correct_action, initializing to zeros.")

        try:
            # Apply scale and offset
            corrected_action = action * self.action_scale + self.action_offset

            # --- Clip action based on provided velocity limits --- #
            # This acts as a safety layer before sending to the robot API.
            # Assuming the action represents velocity (rad/s) or leads to it.
            if self.velocity_limits is not None:
                 corrected_action = np.clip(corrected_action, -self.velocity_limits, self.velocity_limits)
            else:
                 # This case should not happen due to init logic, but handle defensively
                 logger.warning("Velocity limits not available for action clipping.")

            return corrected_action.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during action correction: {e}")
            logger.error(traceback.format_exc())
            return None

    def predict(self, real_state: np.ndarray) -> np.ndarray | None:
        """
        Predict an action suitable for the real robot based on its current state.
        Performs: Real State -> Normalize -> Predict Sim Action -> Correct Action -> Corrected Action.

        Args:
            real_state: State observation from the real robot (must match obs_dim).
                        **CRITICAL:** Ensure this state is correctly formatted by the caller
                        (e.g., `robot_api.get_robot_state_observation()`) and includes any
                        externally calculated relative target/obstacle info if needed.

        Returns:
            np.ndarray | None: The corrected action (matching action_dim) suitable for
                               sending to the robot API, or None if prediction failed.
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return None
        if not self.is_calibrated:
             logger.warning("Predicting using uncalibrated transfer parameters. Results may be inaccurate.")

        # 1. Normalize the real state
        normalized_state = self.normalize_state(real_state)
        if normalized_state is None:
            logger.error("State normalization failed. Cannot predict.")
            return None

        # 2. Get prediction from the RL model
        try:
            # Use deterministic=True for deployment (no exploration noise)
            action_sim, _ = self.model.predict(normalized_state, deterministic=True)
        except Exception as e:
            logger.error(f"Error during model.predict(): {e}")
            logger.error(traceback.format_exc())
            return None

        # 3. Correct the action for the real robot
        corrected_action = self.correct_action(action_sim)
        if corrected_action is None:
            logger.error("Action correction failed. Cannot provide final action.")
            return None

        # logger.debug(f"Predict Cycle: RealState -> NormState -> SimAction -> CorrectedAction")
        # logger.debug(f"  Input State (real): {real_state}")
        # logger.debug(f"  Normalized State:   {normalized_state}")
        # logger.debug(f"  Sim Action:         {action_sim}")
        # logger.debug(f"  Corrected Action:   {corrected_action}")

        return corrected_action 