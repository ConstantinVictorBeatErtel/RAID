"""Build a separate, unadmitted source diagnostic with pinned NVIDIA settings.

No models or datasets are opened here. Existing protocols and collected evidence
are immutable. Structured prompting is deterministic, not a VLM upsampler.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def render_cosmos_prompt(template: str, instruction: str) -> str:
    instruction = instruction.strip().rstrip(".")
    if not instruction:
        raise ValueError("generator instruction must not be empty")
    if not template.lstrip().startswith("{"):
        # Preserve the exact historical plain-text prompt and cache identity.
        return template.format(instruction=instruction)
    payload = json.loads(template)
    if not isinstance(payload, dict) or not isinstance(payload.get("subjects"), list):
        raise ValueError("structured Cosmos prompt requires an object with subjects")

    def replace(value):
        if isinstance(value, str):
            return value.replace("{instruction}", instruction)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    # Substitute after parsing so quotes/newlines in instructions cannot alter JSON.
    return json.dumps(replace(payload), sort_keys=True, ensure_ascii=False)


def structured_prompt_template(height=480, width=480, frames=121, fps=24):
    """Use the pinned model-card schema, with scene details delegated to RGB.

    We do not copy the driving scene from NVIDIA's example or infer privileged
    target positions, future states, or robot actions from demonstration data.
    """
    return json.dumps({
        "subjects": [{
            "description": "The single Franka robot arm visible in the conditioning image",
            "appearance_details": "Preserve the robot's visible geometry, materials and gripper from the input image",
            "relationship": "The robot manipulates the task objects visible in the scene",
            "location": "Start at the location shown in the conditioning image",
            "orientation": "Start with the orientation shown in the conditioning image",
            "pose": "Continue from the visible initial configuration",
            "action": "The robot will {instruction}",
            "state_changes": "Continuous task-related movement and object interaction",
            "number_of_subjects": 1, "number_of_arms": 1, "number_of_legs": 0,
        }],
        "background_setting": "The same simulated tabletop manipulation scene as the input image",
        "lighting": {"conditions": "Preserve the input image's lighting",
                     "direction": "Unchanged", "shadows": "Consistent with the input scene",
                     "illumination_effect": "Stable illumination"},
        "aesthetics": {"composition": "Preserve the input camera composition",
                       "color_scheme": "Preserve visible object colors and materials",
                       "mood_atmosphere": "Robot manipulation simulation", "patterns": ""},
        "cinematography": {"camera_motion": "Stationary camera throughout",
                           "framing": "Same fixed third-person view as the input image",
                           "camera_angle": "Unchanged from the input image",
                           "depth_of_field": "Unchanged", "focus": "Robot and task objects",
                           "lens_focal_length": "Unchanged"},
        "style_medium": "Robotics simulator rendering",
        "artistic_style": "Preserve the visual style of the input simulation",
        "context": "Continue the supplied scene while carrying out: {instruction}",
        "actions": [{"description": "The robot will {instruction} using continuous motion and contact with the relevant objects"}],
        "text_and_signage_elements": [], "segments": [], "transitions": [],
        "temporal_caption": "Starting from the supplied image, the robot will {instruction}. Its articulated structure and object identities remain consistent throughout the motion.",
        "audio_description": "", "resolution": {"W": width, "H": height},
        "aspect_ratio": "1,1", "duration": f"{(frames - 1) / fps:g}s", "fps": fps,
    }, sort_keys=True)


def aligned_protocol(parent: dict) -> dict:
    from .hybrid_protocol import (COSMOS_MODEL_ID, COSMOS_REVISION, DIFFUSERS_COMMIT,
                                  digest, model_digest, validate_hybrid_protocol,
                                  v6_code_snapshot_digest)
    validate_hybrid_protocol(parent)
    original = parent["design"]["generator"]
    if (original["model_id"], original["revision"], original["diffusers_commit"]) != (
            COSMOS_MODEL_ID, COSMOS_REVISION, DIFFUSERS_COMMIT):
        raise ValueError("profile only supports the audited pinned checkpoint and runtime")
    result = copy.deepcopy(parent)
    generator = result["design"]["generator"]
    generator.update(height=480, width=480, num_inference_steps=20,
                     guidance_scale=6.0, flow_shift=12.0,
                     scheduler="UniPCMultistepScheduler", add_resolution_template=False,
                     add_duration_template=False, enable_safety_check=True,
                     prompt_template=structured_prompt_template(), prompt_upsampling=False)
    temporal = result["design"]["temporal"]
    temporal.update(num_frames=121, generation_fps=24, time_scale=1.0,
                    time_scale_status="uncalibrated", timing_basis="nominal_unverified")
    temporal.pop("source_protocol_digest", None)
    result["design"].pop("temporal_calibration_evidence_sha256", None)
    # Old full evidence bundles are tied to the parent scientific identity.
    result["design"]["state_schema"]["parity_status"] = "unverified"
    result["design"].pop("state_parity_evidence_sha256", None)
    result["source_revision"]["hybrid_modules_sha256"] = v6_code_snapshot_digest()
    result["generator_profile"] = {
        "name": "cosmos_edge_pinned_nvidia_square480_v1",
        "parent_protocol_digest": parent["digest"],
        "parent_model_digest": parent["model_digest"],
        "stage": "source_diagnostic", "scientific_admission": False,
        "held_out_claims_eligible": False,
        "reference": f"https://huggingface.co/nvidia/Cosmos3-Edge/blob/{COSMOS_REVISION}/README.md",
        "aspect_ratio_deviation": "480x480 keeps square LIBERO images; NVIDIA's example is 832x480",
        "prompt_method": "deterministic structured template, not image-aware LLM prompt upsampling",
        "timing_note": "24fps maps nominal 8 actions at 20Hz to rounded frame 10; source timing validation required",
    }
    result["model_digest"] = model_digest(result)
    result["digest"] = digest(result)
    validate_hybrid_protocol(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source, destination = Path(args.protocol), Path(args.output)
    if source.resolve() == destination.resolve():
        raise ValueError("write a new protocol; never overwrite the parent")
    updated = aligned_protocol(json.loads(source.read_text()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A candidate is immutable once written. Repeat into a new explicitly named file.
    with destination.open("x") as handle:
        handle.write(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(destination), "digest": updated["digest"],
                      "scientific_admission": False, "held_out_claims_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
