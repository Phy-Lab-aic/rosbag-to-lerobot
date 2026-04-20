"""LeRobot task string + task_type helpers."""

from typing import Dict, Mapping


_CABLE_READABLE = {
    "sfp_sc": "SFP-to-SC",
    "sc_sfp": "SC-to-SFP",
    "sfp": "SFP",
    "sc": "SC",
    "lc": "LC",
}


def _readable_cable(cable_type: str) -> str:
    return _CABLE_READABLE.get(cable_type, cable_type)


def build_task_string(fields: Mapping[str, str]) -> str:
    """Template-based task sentence. Falls back to 'Insert cable.' if any key is empty."""
    cable_type = fields.get("cable_type") or ""
    plug_name = fields.get("plug_name") or ""
    port_name = fields.get("port_name") or ""
    target_module = fields.get("target_module") or ""

    if not (cable_type and plug_name and port_name and target_module):
        return "Insert cable."
    return (
        f"Insert the {_readable_cable(cable_type)} cable's {plug_name}"
        f" into {port_name} on {target_module}."
    )


def build_task_type(fields: Mapping[str, str]) -> str:
    """Short grouping key - 'insert_{cable_type}'."""
    cable_type = fields.get("cable_type") or ""
    return f"insert_{cable_type}" if cable_type else "insert_unknown"
