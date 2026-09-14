#![doc = "PyO3/NumPy boundary for batched native Dodge training."]

use dodge_batch::{
    BatchConfig, BatchEnvironment, BatchError, BatchObservation, ExecutionMode, ObservationFlags,
    PixelBatchObservation, ACTION_COUNT, BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH,
    COLLISION_IMAGE_CONFIG, COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_OBSERVATION_VERSION,
    COLLISION_IMAGE_VALUES, COLLISION_IMAGE_WIDTH, FULL_BOARD_CHANNELS, HAZARD_CHANNELS,
    HAZARD_DEFAULT_HORIZON, HAZARD_DEFAULT_SPAWN_HALO_RADIUS, HAZARD_OBSERVATION_VERSION,
    HAZARD_SCALARS, ML_OBSERVATION_SIZE, PIXEL_HEIGHT, PIXEL_WIDTH,
};
use dodge_core::{Action, FrameEvent, Mode};
use ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

const BATCH_SCHEMA_VERSION: u32 = 1;

/// A persistent native batch environment exposed to Python.
#[pyclass(name = "NativeBatchEnv")]
pub struct NativeBatchEnv {
    inner: BatchEnvironment,
    flags: ObservationFlags,
}

#[pymethods]
impl NativeBatchEnv {
    #[new]
    #[pyo3(signature = (step_frames=4, execution="serial", full_state=false, pixels=false, board=true, difficulty=2, patterns_enabled=true, powerups_enabled=true, include_offscreen_board=false, preserve_offscreen_coordinates=false, ml=false, ml_grid_spacing=32, collision_image=false, enemy_mode="all", permanent_pattern=0, invulnerable=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        step_frames: u32,
        execution: &str,
        full_state: bool,
        pixels: bool,
        board: bool,
        difficulty: u8,
        patterns_enabled: bool,
        powerups_enabled: bool,
        include_offscreen_board: bool,
        preserve_offscreen_coordinates: bool,
        ml: bool,
        ml_grid_spacing: u32,
        collision_image: bool,
        enemy_mode: &str,
        permanent_pattern: u8,
        invulnerable: bool,
    ) -> PyResult<Self> {
        let execution = parse_execution(execution)?;
        let mut config = BatchConfig::new(step_frames);
        config.native.difficulty = difficulty;
        config.native.patterns_enabled = patterns_enabled;
        config.native.powerups_enabled = powerups_enabled;
        config.native.scenario.enemy_mode = match enemy_mode {
            "all" => 0,
            "none" => 1,
            "normal" => 2,
            _ => {
                return Err(PyValueError::new_err(
                    "enemy_mode must be all, none, or normal",
                ))
            }
        };
        if permanent_pattern > 39 || (permanent_pattern != 0 && !patterns_enabled) {
            return Err(PyValueError::new_err(
                "permanent_pattern requires patterns_enabled and an ID in 1..39",
            ));
        }
        config.native.scenario.permanent_pattern = permanent_pattern;
        config.native.scenario.invulnerable = invulnerable;
        config.observations = ObservationFlags {
            full_state,
            pixels,
            board,
            collision_image,
            include_offscreen_board,
            preserve_offscreen_coordinates,
            ml,
            ml_grid_spacing,
        };
        config.execution = execution;
        let inner = BatchEnvironment::new(config).map_err(batch_error)?;
        Ok(Self {
            inner,
            flags: config.observations,
        })
    }

    #[getter]
    fn lane_count(&self) -> usize {
        self.inner.lane_count()
    }

    #[getter]
    fn step_frames(&self) -> u32 {
        self.inner.config().step_frames
    }

    #[getter]
    fn execution(&self) -> &'static str {
        match self.inner.config().execution {
            ExecutionMode::Serial => "serial",
            ExecutionMode::Parallel => "parallel",
        }
    }

    #[getter]
    fn observation_flags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let flags = PyDict::new(py);
        flags.set_item("full_state", self.flags.full_state)?;
        flags.set_item("pixels", self.flags.pixels)?;
        flags.set_item("board", self.flags.board)?;
        flags.set_item("collision_image", self.flags.collision_image)?;
        flags.set_item(
            "include_offscreen_board",
            self.flags.include_offscreen_board,
        )?;
        flags.set_item("ml", self.flags.ml)?;
        flags.set_item(
            "preserve_offscreen_coordinates",
            self.flags.preserve_offscreen_coordinates,
        )?;
        flags.set_item("ml_grid_spacing", self.flags.ml_grid_spacing)?;
        Ok(flags)
    }

    fn reset_batch<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset(&seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_batch_with_startup<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_with_startup(&seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_lanes<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_lanes(&lanes, &seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_lanes_with_startup<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_lanes_with_startup(&lanes, &seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_ml_batch<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml(&seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_batch_with_startup<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml_with_startup(&seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_batch_with_centered_startup<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
        grid_size: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml_with_centered_startup(&seeds, grid_size))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_lanes<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml_lanes(&lanes, &seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_lanes_with_startup<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml_lanes_with_startup(&lanes, &seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_lanes_with_centered_startup<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
        grid_size: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| {
                self.inner
                    .reset_ml_lanes_with_centered_startup(&lanes, &seeds, grid_size)
            })
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step(&native_actions))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn step_batch_active<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
        active: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let native_actions = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let selected = active
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let observations = py
            .detach(|| self.inner.step_active(&native_actions, selected))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    /// Advance lanes through the minimal rendered pixel-only boundary.
    fn step_pixels<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step_pixels(&native_actions))
            .map_err(batch_error)?;
        pixel_observations_to_dict(py, observations)
    }

    fn step_pixels_active<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
        active: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let native_actions = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let selected = active
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let observations = py
            .detach(|| self.inner.step_pixels_active(&native_actions, selected))
            .map_err(batch_error)?;
        pixel_observations_to_dict(py, observations)
    }

    fn step_ml_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step_ml(&native_actions))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    /// Advance lanes while returning only reward, terminal state, and player
    /// position. The full ML observation is reserved for decision boundaries.
    fn step_ml_positions_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step_ml_positions(&native_actions))
            .map_err(batch_error)?;
        ml_position_observations_to_dict(py, observations)
    }

    /// Materialize the current ML observation without advancing any lane.
    fn observe_ml_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let observations = py.detach(|| self.inner.observe_ml()).map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    /// Compute the slow/reference frozen-center hazard field without mutating
    /// the live native lanes.
    #[pyo3(signature = (grid_size, prediction_horizon_frames=32, spawn_halo_radius=1))]
    fn hazard_observations<'py>(
        &self,
        py: Python<'py>,
        grid_size: u32,
        prediction_horizon_frames: u32,
        spawn_halo_radius: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let observations = py
            .detach(|| {
                self.inner.hazard_observations(
                    grid_size,
                    prediction_horizon_frames,
                    spawn_halo_radius,
                )
            })
            .map_err(batch_error)?;
        hazard_observations_to_dict(
            py,
            observations,
            grid_size,
            prediction_horizon_frames,
            spawn_halo_radius,
        )
    }

    /// Return survival-frame deltas for every action from each canonical state.
    /// The live batch is borrowed immutably and is never advanced by scoring.
    fn score_actions<'py>(
        &self,
        py: Python<'py>,
        snapshots: &Bound<'_, PyList>,
        lookahead_steps: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let snapshots = snapshots
            .iter()
            .map(|value| {
                let bytes = value.cast::<PyBytes>().map_err(|_| {
                    PyValueError::new_err(
                        "counterfactual snapshots must be a non-empty list of bytes",
                    )
                })?;
                Ok(bytes.as_bytes().to_vec())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let snapshot_count = snapshots.len();
        let scores = py
            .detach(|| self.inner.score_actions(&snapshots, lookahead_steps))
            .map_err(batch_error)?;
        let values = scores
            .into_iter()
            .flat_map(|row| row.into_iter())
            .collect::<Vec<_>>();
        let scores = Array2::from_shape_vec((snapshot_count, ACTION_COUNT), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py);
        let result = PyDict::new(py);
        result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
        result.set_item("snapshot_count", snapshot_count)?;
        result.set_item("action_count", ACTION_COUNT)?;
        result.set_item("lookahead_steps", lookahead_steps)?;
        result.set_item("scores", scores)?;
        Ok(result)
    }

    #[pyo3(signature = (snapshots, hold_decisions, grid_spacing=32, tolerance=2.0, arrival_latching=false, ban_corner_nodes=false))]
    #[allow(clippy::too_many_arguments)]
    fn score_waypoint_actions<'py>(
        &self,
        py: Python<'py>,
        snapshots: &Bound<'_, PyList>,
        hold_decisions: u32,
        grid_spacing: u32,
        tolerance: f32,
        arrival_latching: bool,
        ban_corner_nodes: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let snapshots = snapshots
            .iter()
            .map(|value| {
                value
                    .cast::<PyBytes>()
                    .map(|bytes| bytes.as_bytes().to_vec())
                    .map_err(|_| {
                        PyValueError::new_err(
                            "waypoint snapshots must be a non-empty list of bytes",
                        )
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let snapshot_count = snapshots.len();
        let scores = py
            .detach(|| {
                self.inner.score_waypoint_actions(
                    &snapshots,
                    hold_decisions,
                    grid_spacing,
                    tolerance,
                    arrival_latching,
                    ban_corner_nodes,
                )
            })
            .map_err(batch_error)?;
        let values = scores.into_iter().flatten().collect::<Vec<_>>();
        let scores = Array2::from_shape_vec((snapshot_count, ACTION_COUNT), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py);
        let result = PyDict::new(py);
        result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
        result.set_item("snapshot_count", snapshot_count)?;
        result.set_item("action_count", ACTION_COUNT)?;
        result.set_item("hold_decisions", hold_decisions)?;
        result.set_item("scores", scores)?;
        Ok(result)
    }
}

#[pymodule]
fn dodge_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(reward_boundary_costs, module)?)?;
    module.add("BATCH_SCHEMA_VERSION", BATCH_SCHEMA_VERSION)?;
    module.add("BOARD_SHAPE", (BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))?;
    module.add(
        "FULL_BOARD_SHAPE",
        (FULL_BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
    )?;
    module.add("PIXEL_SHAPE", (PIXEL_HEIGHT, PIXEL_WIDTH))?;
    module.add(
        "COLLISION_IMAGE_SHAPE",
        (COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_WIDTH),
    )?;
    module.add("COLLISION_IMAGE_VALUES", COLLISION_IMAGE_VALUES)?;
    module.add(
        "COLLISION_IMAGE_OBSERVATION_VERSION",
        COLLISION_IMAGE_OBSERVATION_VERSION,
    )?;
    module.add("COLLISION_IMAGE_CONFIG", COLLISION_IMAGE_CONFIG)?;
    module.add("ML_OBSERVATION_SHAPE", (ML_OBSERVATION_SIZE,))?;
    module.add("HAZARD_CHANNELS", HAZARD_CHANNELS)?;
    module.add("HAZARD_SCALARS", HAZARD_SCALARS)?;
    module.add("HAZARD_OBSERVATION_VERSION", HAZARD_OBSERVATION_VERSION)?;
    module.add("HAZARD_DEFAULT_HORIZON", HAZARD_DEFAULT_HORIZON)?;
    module.add(
        "HAZARD_DEFAULT_SPAWN_HALO_RADIUS",
        HAZARD_DEFAULT_SPAWN_HALO_RADIUS,
    )?;
    module.add_class::<NativeBatchEnv>()?;
    Ok(())
}

fn parse_execution(value: &str) -> PyResult<ExecutionMode> {
    match value {
        "serial" => Ok(ExecutionMode::Serial),
        "parallel" => Ok(ExecutionMode::Parallel),
        _ => Err(PyValueError::new_err(
            "execution must be 'serial' or 'parallel'",
        )),
    }
}

fn action_from_index(index: u8) -> PyResult<Action> {
    Action::ALL
        .get(usize::from(index))
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("invalid Dodge action index: {index}")))
}

fn batch_error(error: BatchError) -> PyErr {
    match error {
        BatchError::Core(core) => PyRuntimeError::new_err(core.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<BatchObservation>,
    flags: ObservationFlags,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let frames = observations.iter().map(|value| value.frame).collect();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let state_hashes = observations.iter().map(|value| value.state_hash).collect();
    let pixel_hashes = observations.iter().map(|value| value.pixel_hash).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let event_flags = observations
        .iter()
        .map(|value| event_flags_code(&value.events))
        .collect();
    let shattered: Vec<u32> = observations.iter().map(|value| value.shattered).collect();
    let score: Vec<f32> = observations.iter().map(|value| value.score).collect();
    let pickups: Vec<u32> = observations
        .iter()
        .map(|value| pickup_count(&value.events))
        .collect();

    let frames = Array1::from_vec(frames).into_pyarray(py);
    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let state_hashes = Array1::from_vec(state_hashes).into_pyarray(py);
    let pixel_hashes = Array1::from_vec(pixel_hashes).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let event_flags = Array1::from_vec(event_flags).into_pyarray(py);
    let shattered = Array1::from_vec(shattered).into_pyarray(py);
    let score = Array1::from_vec(score).into_pyarray(py);

    let pixels = if flags.pixels {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .pixels
                    .as_ref()
                    .into_iter()
                    .flat_map(|pixels| pixels.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array3::from_shape_vec((lane_count, PIXEL_HEIGHT, PIXEL_WIDTH), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };
    let board = if flags.board {
        let board_channels = observations
            .first()
            .and_then(|observation| observation.board.as_ref())
            .map_or(BOARD_CHANNELS, |board| board.shape().0);
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .board
                    .as_ref()
                    .into_iter()
                    .flat_map(|board| board.as_slice().iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array4::from_shape_vec(
                (lane_count, board_channels, BOARD_HEIGHT, BOARD_WIDTH),
                values,
            )
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py),
        )
    } else {
        None
    };
    let collision_image = if flags.collision_image {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .collision_image
                    .as_ref()
                    .into_iter()
                    .flat_map(|image| image.as_slice().iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array3::from_shape_vec(
                (lane_count, COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_WIDTH),
                values,
            )
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py),
        )
    } else {
        None
    };
    let ml_observation = if flags.ml {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .ml_observation
                    .as_ref()
                    .into_iter()
                    .flat_map(|observation| observation.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array2::from_shape_vec((lane_count, ML_OBSERVATION_SIZE), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };
    let player_positions = if flags.ml {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .player_position
                    .as_ref()
                    .into_iter()
                    .flat_map(|position| position.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array2::from_shape_vec((lane_count, 2), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };

    let snapshots = PyList::empty(py);
    for observation in &observations {
        match observation.canonical_snapshot.as_deref() {
            Some(bytes) => snapshots.append(PyBytes::new(py, bytes))?,
            None => snapshots.append(py.None())?,
        }
    }

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    let terms: Vec<f32> = observations
        .iter()
        .flat_map(|value| value.reward_terms)
        .collect();
    let terms = Array2::from_shape_vec((lane_count, 6), terms)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    result.set_item("reward_terms", terms.into_pyarray(py))?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("state_hashes", state_hashes)?;
    result.set_item("pixel_hashes", pixel_hashes)?;
    result.set_item("modes", modes)?;
    result.set_item("event_flags", event_flags)?;
    result.set_item("shattered", shattered)?;
    result.set_item(
        "powerups_collected",
        Array1::from_vec(pickups).into_pyarray(py),
    )?;
    result.set_item("score", score)?;
    result.set_item("pixels", pixels)?;
    result.set_item("board", board)?;
    result.set_item("collision_image", collision_image)?;
    result.set_item(
        "collision_image_version",
        COLLISION_IMAGE_OBSERVATION_VERSION,
    )?;
    result.set_item(
        "collision_image_shape",
        (COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_WIDTH),
    )?;
    result.set_item("collision_image_config", COLLISION_IMAGE_CONFIG)?;
    result.set_item("ml_observation", ml_observation)?;
    result.set_item("player_positions", player_positions)?;
    result.set_item("snapshot_bytes", snapshots)?;
    Ok(result)
}

fn pixel_observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<PixelBatchObservation>,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames = observations.iter().map(|value| value.frame).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let pixels = observations
        .iter()
        .flat_map(|value| value.pixels.iter().copied())
        .collect::<Vec<_>>();
    let player_positions = observations
        .iter()
        .flat_map(|value| value.player_position)
        .collect::<Vec<_>>();

    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames = Array1::from_vec(frames).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let pixels = Array3::from_shape_vec((lane_count, PIXEL_HEIGHT, PIXEL_WIDTH), pixels)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);
    let player_positions = Array2::from_shape_vec((lane_count, 2), player_positions)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("fast_pixels", true)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("modes", modes)?;
    result.set_item("pixels", pixels)?;
    result.set_item("player_positions", player_positions)?;
    Ok(result)
}

fn ml_observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<dodge_batch::MlBatchObservation>,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames = observations.iter().map(|value| value.frame).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let ml_values = observations
        .iter()
        .flat_map(|value| value.ml_observation)
        .collect::<Vec<_>>();
    let positions = observations
        .iter()
        .flat_map(|value| value.player_position)
        .collect::<Vec<_>>();

    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames = Array1::from_vec(frames).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let ml_observation = Array2::from_shape_vec((lane_count, ML_OBSERVATION_SIZE), ml_values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);
    let player_positions = Array2::from_shape_vec((lane_count, 2), positions)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("fast_ml", true)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("modes", modes)?;
    result.set_item("ml_observation", ml_observation)?;
    result.set_item("player_positions", player_positions)?;
    Ok(result)
}

fn ml_position_observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<dodge_batch::MlPositionBatchObservation>,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames = observations.iter().map(|value| value.frame).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let positions = observations
        .iter()
        .flat_map(|value| value.player_position)
        .collect::<Vec<_>>();

    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames = Array1::from_vec(frames).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let player_positions = Array2::from_shape_vec((lane_count, 2), positions)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("fast_ml_positions", true)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("modes", modes)?;
    result.set_item("player_positions", player_positions)?;
    Ok(result)
}

fn hazard_observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<dodge_batch::HazardObservation>,
    grid_size: u32,
    prediction_horizon_frames: u32,
    spawn_halo_radius: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let observation_size =
        HAZARD_CHANNELS * grid_size as usize * grid_size as usize + HAZARD_SCALARS;
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames = observations.iter().map(|value| value.frame).collect();
    let survival_frames = observations
        .iter()
        .map(|value| value.survival_frames)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let hazard_values = observations
        .iter()
        .flat_map(|value| value.hazard_observation.iter().copied())
        .collect::<Vec<_>>();
    let ttc_values = observations
        .iter()
        .flat_map(|value| value.ttc_reference.iter().copied())
        .collect::<Vec<_>>();
    let positions = observations
        .iter()
        .flat_map(|value| value.player_position)
        .collect::<Vec<_>>();

    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames = Array1::from_vec(frames).into_pyarray(py);
    let survival_frames = Array1::from_vec(survival_frames).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let hazard_observation = Array2::from_shape_vec((lane_count, observation_size), hazard_values)
        .map_err(|error| PyRuntimeError::new_err(format!("hazard observation shape: {error}")))?
        .into_pyarray(py);
    let ttc_reference = Array2::from_shape_vec(
        (lane_count, grid_size as usize * grid_size as usize),
        ttc_values,
    )
    .map_err(|error| PyRuntimeError::new_err(format!("hazard TTC shape: {error}")))?
    .into_pyarray(py);
    let player_positions = Array2::from_shape_vec((lane_count, 2), positions)
        .map_err(|error| PyRuntimeError::new_err(format!("hazard positions shape: {error}")))?
        .into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("hazard_observation_version", HAZARD_OBSERVATION_VERSION)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("survival_frames", survival_frames)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("modes", modes)?;
    result.set_item("grid_size", grid_size)?;
    result.set_item("prediction_horizon_frames", prediction_horizon_frames)?;
    result.set_item("spawn_halo_radius", spawn_halo_radius)?;
    result.set_item("hazard_channels", HAZARD_CHANNELS)?;
    result.set_item("hazard_scalars", HAZARD_SCALARS)?;
    result.set_item("hazard_observation", hazard_observation)?;
    result.set_item("ttc_reference", ttc_reference)?;
    result.set_item("player_positions", player_positions)?;
    Ok(result)
}

fn mode_code(mode: Mode) -> u8 {
    match mode {
        Mode::Menu => 0,
        Mode::TransitionToGame => 1,
        Mode::Game => 2,
        Mode::Terminal => 3,
        Mode::Settings => 4,
        Mode::TransitionToSettings => 5,
        Mode::TransitionToMenu => 6,
    }
}

fn event_flags_code(events: &[FrameEvent]) -> u32 {
    events.iter().fold(0_u32, |flags, event| {
        flags
            | match event {
                FrameEvent::EnemySpawn => 1 << 0,
                FrameEvent::Collision => 1 << 1,
                FrameEvent::Death => 1 << 2,
                FrameEvent::PatternActive => 1 << 3,
                FrameEvent::Terminal => 1 << 4,
                FrameEvent::PowerupCollected => 1 << 5,
                FrameEvent::EnemyDestroyed => 1 << 6,
            }
    })
}

#[pyfunction]
fn reward_boundary_costs(
    x: f32,
    y: f32,
    minimum: f32,
    maximum: f32,
    edge_width: f32,
    corner_width: f32,
) -> PyResult<(f32, f32)> {
    dodge_core::reward::boundary_costs(x, y, minimum, maximum, edge_width, corner_width)
        .map(|value| (value.edge, value.corner))
        .ok_or_else(|| PyValueError::new_err("invalid native reward boundary geometry"))
}

fn pickup_count(events: &[FrameEvent]) -> u32 {
    events
        .iter()
        .filter(|event| **event == FrameEvent::PowerupCollected)
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::{action_from_index, event_flags_code, mode_code, NativeBatchEnv};
    use dodge_core::{Action, FrameEvent, Mode};

    #[test]
    fn v70_pickup_count_preserves_multiplicity_and_legacy_bits() {
        let events = [
            FrameEvent::PowerupCollected,
            FrameEvent::Death,
            FrameEvent::PowerupCollected,
        ];
        assert_eq!(super::pickup_count(&events), 2);
        assert_eq!(event_flags_code(&events), (1 << 5) | (1 << 2));
        assert_eq!(super::pickup_count(&[]), 0);
    }

    #[test]
    fn action_indices_preserve_the_nine_action_contract() {
        assert_eq!(action_from_index(0).ok(), Some(Action::Neutral));
        assert_eq!(action_from_index(8).ok(), Some(Action::DownRight));
        assert!(action_from_index(9).is_err());
    }

    #[test]
    fn metadata_codes_are_stable() {
        assert_eq!(mode_code(Mode::Menu), 0);
        assert_eq!(mode_code(Mode::Game), 2);
        assert_eq!(
            event_flags_code(&[FrameEvent::EnemySpawn, FrameEvent::Terminal]),
            17
        );
    }

    #[test]
    fn collision_image_trailing_constructor_flag_is_stable() {
        let environment = NativeBatchEnv::new(
            4, "serial", false, false, false, 2, true, true, false, false, false, 32, true, "all",
            0, false,
        )
        .unwrap_or_else(|_| unreachable!("valid native constructor configuration"));
        assert!(environment.flags.collision_image);
    }
}
