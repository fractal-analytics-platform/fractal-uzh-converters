"""Utility functions for MD ImageXpress HCS.ai data."""

import json
import logging
import math
from pathlib import Path

import pandas as pd
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AcquisitionOptions,
    AttributeType,
    ChannelInfo,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    default_axes_builder,
    filesystem_for_url,
    join_url_paths,
    tiles_aggregation_pipeline,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fractal_uzh_converters.common import (
    STANDARD_ROWS_NAMES,
    HCSBaseAcquisitionModel,
    clean_channel_string,
    get_attributes_from_condition_table,
)

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Model
#
######################################################################


class MDAcquisitionOptions(AcquisitionOptions):
    """Acquisition options for conversion."""

    convert_only_projections: bool = Field(default=False)
    """
    If True, only convert projection images, if available.
    """
    convert_montages: bool = Field(default=False)
    """
    If True, convert montaged / stitched images, if available.
    """


class MDImageXpressHCSaiAcquisitionModel(HCSBaseAcquisitionModel):
    """Acquisition details for the MD ImageXpress HCS.ai microscope data."""

    advanced: MDAcquisitionOptions = Field(default_factory=MDAcquisitionOptions)
    """
    Advanced acquisition options.
    """

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Make the path more flexible.

        Allow:
         - path/to/acquisition/{protocol}.mxprotocol
         - path/to/acquisition/
        """
        v = v.rstrip("/")
        if v.endswith(".mxprotocol"):
            # Strip the filename to get the directory
            return str(Path(v).parent)
        return v


######################################################################
#
# Pydantic models for parsing MD ImageXpress metadata
#
######################################################################


class MDFilterInfo(BaseModel):
    """Emission or excitation filter information from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = Field(..., alias="Name")
    wavelength: float = Field(..., alias="Wavelength")
    unit: str = Field(default="nm", alias="Unit")


class MDWavelength(BaseModel):
    """Wavelength/channel configuration from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    index: int = Field(..., alias="Index")
    imaging_mode: str = Field(..., alias="ImagingMode")
    z_slice: int = Field(..., alias="ZSlice")
    z_step: float = Field(..., alias="ZStep")
    emission_filter: MDFilterInfo = Field(..., alias="EmissionFilter")
    excitation_filter: MDFilterInfo = Field(..., alias="ExcitationFilter")


class MDCameraSize(BaseModel):
    """Camera sensor size in pixels from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    width: int = Field(..., alias="Width")
    height: int = Field(..., alias="Height")


class MDCamera(BaseModel):
    """Camera configuration from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    size: MDCameraSize = Field(..., alias="Size")
    binning: str = Field(..., alias="Binning")


class MDObjectiveCalibration(BaseModel):
    """Objective calibration data from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    unit: str = Field(..., alias="Unit")
    objective_name: str = Field(..., alias="ObjectiveName")
    pixel_width: float = Field(..., alias="PixelWidth")
    pixel_height: float = Field(..., alias="PixelHeight")


class MDZDimensionParameters(BaseModel):
    """Z dimension parameters from JDCE PlateMap."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = Field(..., alias="Enabled")
    step: float = Field(..., alias="Step")
    number_of_slices: int = Field(..., alias="NumberOfSlices")
    variable: bool = Field(..., alias="Variable")


class MDTimePoint(BaseModel):
    """Single time point entry from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ms: int = Field(..., alias="Ms")


class MDTimeSchedule(BaseModel):
    """Time schedule from JDCE PlateMap."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = Field(..., alias="Enabled")
    number_of_timepoints: int = Field(..., alias="NumberOfTimepoints")
    times: list[MDTimePoint] = Field(..., alias="Times")
    variable: bool = Field(..., alias="Variable")


class MDPlateMap(BaseModel):
    """PlateMap section from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    z_dimension_parameters: MDZDimensionParameters = Field(
        ..., alias="ZDimensionParameters"
    )
    time_schedule: MDTimeSchedule = Field(..., alias="TimeSchedule")


class MDProtocol(BaseModel):
    """AutoLeadAcquisitionProtocol section from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    camera: MDCamera = Field(..., alias="Camera")
    objective_calibration: MDObjectiveCalibration = Field(
        ..., alias="ObjectiveCalibration"
    )
    wavelengths: list[MDWavelength] = Field(..., alias="Wavelengths")
    plate_map: MDPlateMap = Field(..., alias="PlateMap")


class MDImageStack(BaseModel):
    """ImageStack section from JDCE."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    plate_id: str = Field(..., alias="PlateId")
    uuid: str = Field(..., alias="Uuid")
    image_format: str = Field(..., alias="ImageFormat")
    large_image: bool = Field(..., alias="LargeImage")
    auto_lead_acquisition_protocol: MDProtocol = Field(
        ..., alias="AutoLeadAcquisitionProtocol"
    )


class MDExperimentMeta(BaseModel):
    """Top-level experiment metadata from JDCE file.

    Analogous to CQ3K's MeasurementDetail. Captures the acquisition-level
    metadata from the JDCE JSON file.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    version: str = Field(..., alias="Version")
    image_stack: MDImageStack = Field(..., alias="ImageStack")

    @property
    def protocol(self) -> MDProtocol:
        """Shortcut to the acquisition protocol."""
        return self.image_stack.auto_lead_acquisition_protocol

    @property
    def pixel_size_x(self) -> float:
        """Pixel width in micrometers."""
        return self.protocol.objective_calibration.pixel_width

    @property
    def pixel_size_y(self) -> float:
        """Pixel height in micrometers."""
        return self.protocol.objective_calibration.pixel_height

    @property
    def z_step_um(self) -> float:
        """Z step in micrometers. Returns 1.0 if step is 0."""
        step = self.protocol.plate_map.z_dimension_parameters.step
        return step if step != 0.0 else 1.0

    @property
    def is_time_series(self) -> bool:
        """Whether this is a time series acquisition."""
        return len(self.protocol.plate_map.time_schedule.times) > 1

    @property
    def image_width_px(self) -> int:
        """Image width in pixels."""
        return self.protocol.camera.size.width

    @property
    def image_height_px(self) -> int:
        """Image height in pixels."""
        return self.protocol.camera.size.height

    @property
    def channels(self) -> list[MDWavelength]:
        """List of wavelength/channel configurations."""
        return self.protocol.wavelengths


class MDImageRecord(BaseModel):
    """Per-image metadata record from MD ImageXpress CSV.

    Each row in the CSV file represents a single acquired image tile.
    Analogous to OperettaImageMeta and CQ3K's ImageMeasurementRecord.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # === Identifiers (essential) ===
    row: int = Field(..., alias="Row")
    column: int = Field(..., alias="Column")
    field: int = Field(..., alias="Field")
    wavelength: int = Field(..., alias="Wavelength")
    timepoint: int = Field(..., alias="Timepoint")
    z_index: int = Field(..., alias="ZIndex")

    # === Image dimensions (essential) ===
    image_size_x_px: int = Field(..., alias="ImageSizeXPx")
    image_size_y_px: int = Field(..., alias="ImageSizeYPx")

    # === File location (essential) ===
    image_sub_folder_path: str = Field(..., alias="ImageSubFolderPath")
    image_file_name: str = Field(..., alias="ImageFileName")

    # === Spatial positioning (essential) ===
    field_offset_point_x: float = Field(..., alias="FieldOffsetPointX")
    field_offset_point_y: float = Field(..., alias="FieldOffsetPointY")

    # === Optional identifiers ===
    fov_uuid: str | None = Field(default=None, alias="FovUuid")
    well: str | None = Field(default=None, alias="Well")

    # === Optional pixel offsets ===
    image_start_x_px: int | None = Field(default=None, alias="ImageStartXPx")
    image_start_y_px: int | None = Field(default=None, alias="ImageStartYPx")

    # === Optional timing ===
    timestamp_sec: float | None = Field(default=None, alias="TimeStampSec")
    exposure_time_ms: float | None = Field(default=None, alias="ExposureTimeMs")

    # === Optional detection ===
    excitation_emission_filter: str | None = Field(
        default=None, alias="ExcitationEmissionFilter"
    )
    min_intensity: float | None = Field(default=None, alias="MinIntensity")
    max_intensity: float | None = Field(default=None, alias="MaxIntensity")
    mean_intensity: float | None = Field(default=None, alias="MeanIntensity")

    # === Optional absolute positions ===
    position_x_um: float | None = Field(default=None, alias="PositionXUm")
    position_y_um: float | None = Field(default=None, alias="PositionYUm")
    position_z_um: float | None = Field(default=None, alias="PositionZUm")

    # === Optional environmental ===
    temperature_c: float | None = Field(default=None, alias="TemperatureC")
    co2: float | None = Field(default=None, alias="CO2")
    o2_level: float | None = Field(default=None, alias="O2Level")

    # === Optional metadata ===
    annotations: str | None = Field(default=None, alias="Annotations")
    checksum: str | None = Field(default=None, alias="Checksum")

    @field_validator(
        "fov_uuid",
        "well",
        "excitation_emission_filter",
        "annotations",
        "checksum",
        mode="before",
    )
    @classmethod
    def nan_to_none_str(cls, v: object) -> str | None:
        """Convert NaN/empty values to None for string fields."""
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return str(v)

    @field_validator(
        "timestamp_sec",
        "exposure_time_ms",
        "min_intensity",
        "max_intensity",
        "mean_intensity",
        "position_x_um",
        "position_y_um",
        "position_z_um",
        "temperature_c",
        "co2",
        "o2_level",
        mode="before",
    )
    @classmethod
    def nan_to_none_float(cls, v: object) -> float | None:
        """Convert NaN values to None for float fields."""
        if v is None:
            return None
        if isinstance(v, float):
            return None if math.isnan(v) else v
        return float(str(v))

    @property
    def row_letter(self) -> str:
        """Convert 1-based row index to letter (1 -> 'A', 2 -> 'B', etc.)."""
        return STANDARD_ROWS_NAMES[self.row - 1]

    @property
    def well_id(self) -> str:
        """Get well ID in format 'A01'."""
        return f"{self.row_letter}{self.column:02d}"

    @property
    def fov_name(self) -> str:
        """Get FOV name in format 'FOV0', 'FOV1', etc."""
        return f"FOV{self.field}"

    @property
    def relative_image_path(self) -> str:
        """Get relative path to the image file."""
        return f"{self.image_sub_folder_path}/{self.image_file_name}"


######################################################################
#
# Load MD metadata
#
######################################################################


def parse_jdce_metadata(file_path: str) -> MDExperimentMeta:
    """Parse JDCE metadata file into a structured model."""
    fs = filesystem_for_url(file_path)
    with fs.open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return MDExperimentMeta.model_validate(data)


def load_csv_metadata(file_path: str) -> list[MDImageRecord]:
    """Load CSV metadata file into structured records."""
    fs = filesystem_for_url(file_path)
    with fs.open(file_path) as f:
        df = pd.read_csv(f)
    return [MDImageRecord.model_validate(row.to_dict()) for _, row in df.iterrows()]


######################################################################
#
# Helper functions for building tiles (following Operetta pattern)
#
######################################################################


def _build_acquisition_details(
    experiment_meta: MDExperimentMeta,
    acquisition_model: MDImageXpressHCSaiAcquisitionModel,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from experiment metadata."""
    channels = []
    for wl in experiment_meta.channels:
        if wl.index != len(channels):
            raise ValueError(
                f"Channel index mismatch: expected {len(channels)}, got {wl.index}"
            )
        wavelength_id = str(int(wl.emission_filter.wavelength))
        channels.append(
            ChannelInfo(
                # The wavelength is a required numeric field, so its id is never
                # blank and serves as the fallback for an unnamed filter.
                channel_label=(
                    clean_channel_string(wl.emission_filter.name) or wavelength_id
                ),
                wavelength_id=wavelength_id,
            )
        )
    acq = AcquisitionDetails(
        channels=channels,
        xy_pixel_size=experiment_meta.pixel_size_x,
        z_spacing=experiment_meta.z_step_um,
        t_spacing=1.0,
        axes=default_axes_builder(is_time_series=experiment_meta.is_time_series),
        start_x_space="world",
        start_y_space="world",
        start_z_space="pixel",
        start_t_space="pixel",
        length_x_space="pixel",
        length_y_space="pixel",
        length_z_space="pixel",
        length_t_space="pixel",
    )
    return acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acq
    )


def _build_tiles(
    images: list[MDImageRecord],
    experiment_dir: str,
    experiment_meta: MDExperimentMeta,
    acquisition_model: MDImageXpressHCSaiAcquisitionModel,
    row_letter: str,
    column: int,
    fov_idx: int,
    attributes: dict[str, AttributeType],
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    acquisition_details = _build_acquisition_details(experiment_meta, acquisition_model)
    image_in_plate = ImageInPlate(
        plate_name=acquisition_model.normalized_plate_name,
        row=row_letter,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )
    fov_name = f"FOV_{fov_idx}"
    tiles = []
    for img in images:
        tiff_path = join_url_paths(experiment_dir, img.relative_image_path)
        tiles.append(
            Tile(
                fov_name=fov_name,
                start_x=img.field_offset_point_x,
                length_x=img.image_size_x_px,
                start_y=img.field_offset_point_y,
                length_y=img.image_size_y_px,
                start_z=img.z_index,
                length_z=1,
                start_c=img.wavelength,
                length_c=1,
                start_t=img.timepoint,
                length_t=1,
                collection=image_in_plate,
                image_loader=DefaultImageLoader(file_path=tiff_path),
                acquisition_details=acquisition_details,
                attributes=attributes,
            )
        )
    return tiles


######################################################################
#
# Main metadata parsing function
#
######################################################################


def parse_md_metadata(
    *,
    acquisition_model: MDImageXpressHCSaiAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse MD ImageXpress HCS.ai metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    root_dir = acquisition_model.path

    # Discover available experiment directories
    available_dirs = {
        "montage": list(Path(root_dir).glob("experiment_montage")),
        "z_stack": list(Path(root_dir).glob("experiment_z_stack")),
        "standard": list(Path(root_dir).glob("experiment")),
    }

    # Check if any experiment directories exist
    if not any(available_dirs.values()):
        raise FileNotFoundError(
            f"No 'experiment*' folders found in {root_dir}. Please ensure the path is "
            "correct and contains the expected folder structure."
        )

    # Select the appropriate directory based on conversion options
    if acquisition_model.advanced.convert_montages:
        if not available_dirs["montage"]:
            available = [k for k, v in available_dirs.items() if v]
            raise FileNotFoundError(
                f"No 'experiment_montage' folder found in {root_dir} for montages. "
                f"Available folders: {available}. "
                "Hint: Disable the 'convert_montages' option."
            )
        experiment_dir = available_dirs["montage"][0]
    elif not acquisition_model.advanced.convert_only_projections:
        if available_dirs["z_stack"]:
            experiment_dir = available_dirs["z_stack"][0]
        elif available_dirs["standard"]:
            experiment_dir = available_dirs["standard"][0]
        else:
            available = [k for k, v in available_dirs.items() if v]
            raise FileNotFoundError(
                f"No 'experiment_z_stack' or 'experiment' folder found in {root_dir}. "
                f"Available folders: {available}."
            )
    else:
        if not available_dirs["standard"]:
            available = [k for k, v in available_dirs.items() if v]
            raise FileNotFoundError(
                f"No 'experiment' folder found in {root_dir} for projections. "
                f"Available folders: {available}. "
                "Hint: Disable the 'convert_only_projections' option."
            )
        experiment_dir = available_dirs["standard"][0]

    condition_table = acquisition_model.get_condition_table()

    # Load experiment metadata from .jdce file
    jdce_files = sorted(Path(experiment_dir).glob("*.jdce"))
    if len(jdce_files) == 0:
        raise FileNotFoundError(f"No .jdce file found in directory: {experiment_dir}")
    elif len(jdce_files) > 1:
        raise ValueError(
            f"Multiple .jdce files found in directory: {experiment_dir}."
            "Please ensure there is only one .jdce file."
        )
    experiment_meta = parse_jdce_metadata(str(jdce_files[0]))

    # Load image records from .csv file
    csv_files = sorted(Path(experiment_dir).glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .csv file found in directory: {experiment_dir}")
    elif len(csv_files) > 1:
        raise ValueError(
            f"Multiple .csv files found in directory: {experiment_dir}."
            "Please ensure there is only one .csv file."
        )
    records = load_csv_metadata(str(csv_files[0]))

    # Determine if data is a z-stack by checking for multiple Z indices
    is_z_stack = len({r.z_index for r in records}) > 1

    # Check if data is compatible with projection and montage conversion options
    if (
        acquisition_model.advanced.convert_only_projections
        and acquisition_model.advanced.convert_montages
        and is_z_stack
    ):
        raise ValueError(
            "Both convert_only_projections and convert_montages are True, but the "
            "montage-data is a z-stack. "
            "Hint: Set either convert_only_projections or convert_montages to False."
        )

    # Group records by well + FOV
    groups: dict[tuple[int, int, int], list[MDImageRecord]] = {}
    for rec in records:
        key = (rec.row, rec.column, rec.field)
        groups.setdefault(key, []).append(rec)

    # Build tiles for each group
    all_tiles: list[Tile] = []
    for (row, column, fov_idx), images in groups.items():
        row_letter = STANDARD_ROWS_NAMES[row - 1]
        attributes = get_attributes_from_condition_table(
            condition_table=condition_table,
            row=row_letter,
            column=column,
            acquisition=acquisition_model.acquisition_id,
        )
        tiles = _build_tiles(
            images=images,
            experiment_dir=str(experiment_dir),
            experiment_meta=experiment_meta,
            acquisition_model=acquisition_model,
            row_letter=row_letter,
            column=column,
            fov_idx=fov_idx,
            attributes=attributes,
        )
        all_tiles.extend(tiles)

    logger.info(f"Built {len(all_tiles)} tiles from {root_dir}")

    tiled_images = tiles_aggregation_pipeline(
        tiles=all_tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
    )

    return tiled_images
