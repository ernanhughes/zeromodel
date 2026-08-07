"""Deterministic panel-observation evidence compiler.

This compiler is intentionally simple and bounded to the declared panel layout.
It is a stage-2 scaffold for real camera captures: no OCR, no learned model, and
no confident value when measurements are ambiguous.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from zeromodel.perception import (
    EvidenceCompilationReport,
    FieldEvidence,
    FieldMeasurement,
    ObservationArtifact,
    ObservationValidityReport,
    PanelRegistrationResult,
)

ROOT = Path(__file__).resolve().parent
REGIONS_PATH = ROOT / "regions.json"
CALIBRATION_PATH = ROOT / "calibration.json"
COMPILER_ID = "fixed-camera-panel-evidence-compiler/v1"
REGISTRATION_METHOD_ID = "anchor-threshold-canonical-panel/v1"


@dataclass(frozen=True, slots=True)
class PanelObservationEvidenceCompiler:
    regions: Mapping[str, object]
    calibration: Mapping[str, object]

    @classmethod
    def load_default(cls) -> "PanelObservationEvidenceCompiler":
        return cls(
            regions=json.loads(REGIONS_PATH.read_text(encoding="utf-8")),
            calibration=json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")),
        )

    @property
    def panel_layout_id(self) -> str:
        return str(self.regions["panel_layout_id"])

    @property
    def calibration_id(self) -> str:
        return str(self.calibration["calibration_id"])

    def compile_image(
        self,
        image_path: Path,
        *,
        observation_id: str | None = None,
        capture_timestamp: str = "fixture",
        camera_id: str = "unknown-camera",
        capture_profile_id: str = "unknown-profile",
    ) -> tuple[ObservationArtifact, EvidenceCompilationReport]:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        digest = f"sha256:{hashlib.sha256(image_path.read_bytes()).hexdigest()}"
        observation = ObservationArtifact(
            observation_id=observation_id or digest,
            image_digest=digest,
            capture_timestamp=capture_timestamp,
            camera_id=camera_id,
            capture_profile_id=capture_profile_id,
            width=image.width,
            height=image.height,
        )
        validity = self._validity(observation.observation_id, array)
        registration = self._registration(observation.observation_id, array)
        if not validity.valid or not registration.valid:
            reason = ";".join(validity.invalid_reasons) or registration.reason
            evidence = (
                FieldEvidence(
                    field_id="panel",
                    status="invalid_observation",
                    supported_values=(),
                    contradicted_values=(),
                    unresolved_values=(),
                    source_region="panel",
                    observation_id=observation.observation_id,
                    compiler_id=COMPILER_ID,
                    registered_panel_id=registration.registered_panel_id,
                    decoder_id="observation-validity/v1",
                    raw_measurement=validity.measurement,
                    reason=reason,
                ),
            )
            return observation, EvidenceCompilationReport(
                observation_id=observation.observation_id,
                panel_layout_id=self.panel_layout_id,
                calibration_id=self.calibration_id,
                compiler_id=COMPILER_ID,
                validity_report=validity,
                registration_result=registration,
                measurements=(),
                evidence=evidence,
            )

        measurements: list[FieldMeasurement] = []
        evidence_items: list[FieldEvidence] = []
        for field_id in ("power", "mode", "temperature", "door", "alarm"):
            measurement, evidence = self._compile_field(
                field_id,
                observation.observation_id,
                registration.registered_panel_id,
                array,
            )
            measurements.append(measurement)
            evidence_items.append(evidence)
        return observation, EvidenceCompilationReport(
            observation_id=observation.observation_id,
            panel_layout_id=self.panel_layout_id,
            calibration_id=self.calibration_id,
            compiler_id=COMPILER_ID,
            validity_report=validity,
            registration_result=registration,
            measurements=tuple(measurements),
            evidence=tuple(evidence_items),
        )

    def _validity(
        self, observation_id: str, array: np.ndarray
    ) -> ObservationValidityReport:
        brightness = float(np.mean(array))
        glare_fraction = float(np.mean(np.all(array >= 246, axis=2)))
        invalid_reasons: list[str] = []
        if brightness < float(self.calibration["minimum_valid_brightness"]):
            invalid_reasons.append("too_dark")
        if brightness > float(self.calibration["maximum_valid_brightness"]):
            invalid_reasons.append("too_bright")
        if glare_fraction > float(self.calibration["maximum_glare_fraction"]):
            invalid_reasons.append("excessive_glare")
        return ObservationValidityReport(
            observation_id=observation_id,
            valid=not invalid_reasons,
            invalid_reasons=tuple(invalid_reasons),
            measurement={
                "mean_brightness": round(brightness, 6),
                "glare_fraction": round(glare_fraction, 6),
            },
        )

    def _registration(
        self, observation_id: str, array: np.ndarray
    ) -> PanelRegistrationResult:
        anchors = self.regions["anchors"]  # type: ignore[index]
        threshold = float(self.calibration["minimum_anchor_dark_fraction"])
        found: list[tuple[float, float]] = []
        for region in anchors.values():  # type: ignore[union-attr]
            crop = self._crop(array, region)
            dark_fraction = float(np.mean(np.mean(crop, axis=2) < 45.0))
            if dark_fraction >= threshold:
                x0, y0, x1, y1 = self._rect(region)
                found.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        valid = len(found) == 4
        registered_panel_id = f"sha256:{hashlib.sha256((observation_id + self.panel_layout_id).encode()).hexdigest()}"
        return PanelRegistrationResult(
            observation_id=observation_id,
            panel_layout_id=self.panel_layout_id,
            registration_method_id=REGISTRATION_METHOD_ID,
            registered_panel_id=registered_panel_id,
            valid=valid,
            anchor_count=len(found),
            source_corners=tuple(found),
            reason="" if valid else "missing_panel_anchors",
        )

    def _compile_field(
        self,
        field_id: str,
        observation_id: str,
        registered_panel_id: str,
        array: np.ndarray,
    ) -> tuple[FieldMeasurement, FieldEvidence]:
        if field_id == "power":
            raw, supported, reason = self._decode_power(array)
            decoder_id = "colour-led/v1"
        elif field_id == "mode":
            raw, supported, reason = self._decode_mode(array)
            decoder_id = "position-marker/v1"
        elif field_id == "temperature":
            raw, supported, reason = self._decode_temperature(array)
            decoder_id = "three-segment-bar/v1"
        elif field_id == "door":
            raw, supported, reason = self._decode_door(array)
            decoder_id = "door-geometry/v1"
        elif field_id == "alarm":
            raw, supported, reason = self._decode_alarm(array)
            decoder_id = "diamond-fill/v1"
        else:
            raise ValueError(field_id)

        field = self.regions["fields"][field_id]  # type: ignore[index]
        allowed = tuple(field["allowed_values"])  # type: ignore[index]
        region_id = str(field["region_id"])  # type: ignore[index]
        measurement = FieldMeasurement(
            field_id=field_id,
            observation_id=observation_id,
            registered_panel_id=registered_panel_id,
            source_region=region_id,
            decoder_id=decoder_id,
            raw_measurement=raw,
        )
        if supported:
            evidence = FieldEvidence(
                field_id=field_id,
                status="supported",
                supported_values=(supported,),
                contradicted_values=tuple(
                    value for value in allowed if value != supported
                ),
                unresolved_values=(),
                source_region=region_id,
                observation_id=observation_id,
                compiler_id=COMPILER_ID,
                registered_panel_id=registered_panel_id,
                decoder_id=decoder_id,
                raw_measurement=raw,
            )
        else:
            evidence = FieldEvidence(
                field_id=field_id,
                status="unresolved",
                supported_values=(),
                contradicted_values=(),
                unresolved_values=allowed,
                source_region=region_id,
                observation_id=observation_id,
                compiler_id=COMPILER_ID,
                registered_panel_id=registered_panel_id,
                decoder_id=decoder_id,
                raw_measurement=raw,
                reason=reason,
            )
        return measurement, evidence

    def _decode_power(self, array: np.ndarray) -> tuple[dict[str, object], str, str]:
        crop = self._field_crop(array, "power", "led")
        rgb = np.mean(crop.reshape(-1, 3), axis=0) / 255.0
        red, green, blue = (float(value) for value in rgb)
        saturation = float(max(rgb) - min(rgb))
        raw = {
            "mean_red": round(red, 6),
            "mean_green": round(green, 6),
            "mean_blue": round(blue, 6),
            "mean_saturation": round(saturation, 6),
        }
        if saturation < float(self.calibration["minimum_colour_saturation"]):
            return raw, "off", ""
        if green > red + 0.18 and green > blue + 0.18:
            return raw, "green", ""
        if red > green + 0.18 and red > blue + 0.18:
            return raw, "red", ""
        return raw, "", "ambiguous_led_colour"

    def _decode_alarm(self, array: np.ndarray) -> tuple[dict[str, object], str, str]:
        crop = self._field_crop(array, "alarm", "diamond")
        red_fraction = self._colour_fraction(crop, red=True)
        dark_fraction = float(np.mean(np.mean(crop, axis=2) < 80.0))
        raw = {
            "red_fraction": round(red_fraction, 6),
            "dark_fraction": round(dark_fraction, 6),
        }
        if red_fraction >= 0.20:
            return raw, "active", ""
        if dark_fraction >= 0.15:
            return raw, "inactive", ""
        return raw, "", "ambiguous_alarm_fill"

    def _decode_mode(self, array: np.ndarray) -> tuple[dict[str, object], str, str]:
        boxes = self.regions["fields"]["mode"]["boxes"]  # type: ignore[index]
        occupancies = {
            value: self._blue_fraction(self._crop(array, region))
            for value, region in boxes.items()  # type: ignore[union-attr]
        }
        winner, winner_value = max(occupancies.items(), key=lambda item: item[1])
        runner_up = max(value for key, value in occupancies.items() if key != winner)
        margin = winner_value - runner_up
        raw = {
            **{
                f"{key}_occupancy": round(value, 6)
                for key, value in occupancies.items()
            },
            "winner": winner,
            "winner_margin": round(margin, 6),
        }
        if winner_value >= float(
            self.calibration["minimum_marker_occupancy"]
        ) and margin >= float(self.calibration["minimum_marker_margin"]):
            return raw, winner, ""
        return raw, "", "ambiguous_mode_marker"

    def _decode_temperature(
        self, array: np.ndarray
    ) -> tuple[dict[str, object], str, str]:
        segments = self.regions["fields"]["temperature"]["segments"]  # type: ignore[index]
        occupancies = {
            value: self._filled_fraction(self._crop(array, region))
            for value, region in segments.items()  # type: ignore[union-attr]
        }
        active = [
            value
            for value in ("normal", "elevated", "critical")
            if occupancies[value] >= float(self.calibration["minimum_bar_occupancy"])
        ]
        raw = {
            **{
                f"{key}_occupancy": round(value, 6)
                for key, value in occupancies.items()
            },
            "active_count": len(active),
        }
        if active == ["normal"]:
            return raw, "normal", ""
        if active == ["normal", "elevated"]:
            return raw, "elevated", ""
        if active == ["normal", "elevated", "critical"]:
            return raw, "critical", ""
        return raw, "", "ambiguous_temperature_bar"

    def _decode_door(self, array: np.ndarray) -> tuple[dict[str, object], str, str]:
        field = self.regions["fields"]["door"]  # type: ignore[index]
        closed = self._blue_fraction(self._crop(array, field["closed_region"]))  # type: ignore[index]
        open_ = self._blue_fraction(self._crop(array, field["open_region"]))  # type: ignore[index]
        margin = abs(closed - open_)
        raw = {
            "closed_occupancy": round(closed, 6),
            "open_occupancy": round(open_, 6),
            "winner_margin": round(margin, 6),
        }
        minimum = float(self.calibration["minimum_shape_occupancy"])
        required_margin = float(self.calibration["minimum_shape_margin"])
        if closed >= minimum and closed - open_ >= required_margin:
            return raw, "closed", ""
        if open_ >= minimum and open_ - closed >= required_margin:
            return raw, "open", ""
        return raw, "", "ambiguous_door_geometry"

    def _field_crop(self, array: np.ndarray, field_id: str, key: str) -> np.ndarray:
        return self._crop(array, self.regions["fields"][field_id][key])  # type: ignore[index]

    @staticmethod
    def _rect(region: object) -> tuple[int, int, int, int]:
        values = tuple(int(value) for value in region)  # type: ignore[arg-type]
        return values  # type: ignore[return-value]

    def _crop(self, array: np.ndarray, region: object) -> np.ndarray:
        x0, y0, x1, y1 = self._rect(region)
        return array[y0:y1, x0:x1, :]

    @staticmethod
    def _blue_fraction(crop: np.ndarray) -> float:
        red = crop[:, :, 0].astype(np.int16)
        green = crop[:, :, 1].astype(np.int16)
        blue = crop[:, :, 2].astype(np.int16)
        return float(np.mean((blue > red + 35) & (blue > green + 20) & (blue > 120)))

    @staticmethod
    def _colour_fraction(crop: np.ndarray, *, red: bool) -> float:
        r = crop[:, :, 0].astype(np.int16)
        g = crop[:, :, 1].astype(np.int16)
        b = crop[:, :, 2].astype(np.int16)
        if red:
            return float(np.mean((r > g + 40) & (r > b + 40) & (r > 120)))
        return float(np.mean((g > r + 40) & (g > b + 40) & (g > 120)))

    @staticmethod
    def _filled_fraction(crop: np.ndarray) -> float:
        channel_max = np.max(crop, axis=2)
        channel_min = np.min(crop, axis=2)
        return float(np.mean((channel_max - channel_min > 40) & (channel_max > 120)))
