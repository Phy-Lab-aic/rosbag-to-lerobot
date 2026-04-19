from v3_conversion.aic_meta.task_string import build_task_string, build_task_type


def test_build_task_string_uses_readable_cable_type():
    fields = {
        "cable_type": "sfp_sc",
        "plug_name": "sfp_tip",
        "port_name": "sfp_port_0",
        "target_module": "nic_card_mount_0",
    }
    assert build_task_string(fields) == (
        "Insert the SFP-to-SC cable's sfp_tip into sfp_port_0 on nic_card_mount_0."
    )


def test_build_task_string_falls_back_when_missing():
    assert build_task_string({"cable_type": ""}) == "Insert cable."


def test_build_task_type_snake_case():
    assert build_task_type({"cable_type": "sfp_sc"}) == "insert_sfp_sc"
    assert build_task_type({"cable_type": ""}) == "insert_unknown"
