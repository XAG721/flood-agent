from __future__ import annotations


def test_package_public_exports_remain_available_lazily():
    from flood_system import FloodWarningSystem, app, create_app, create_default_system

    assert app.title == "Flood Warning System"
    assert FloodWarningSystem.__name__ == "FloodWarningSystem"
    assert callable(create_app)
    assert callable(create_default_system)
