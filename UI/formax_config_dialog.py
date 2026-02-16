from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "_Miscell" / "ConnectorConfigs"


DEFAULT_FORMAX = {
    "empty_i0": 1.0,
    "empty_it": 1.0,
    "qSumStart": 0.0,
    "qSumEnd": 0.0,
    "absolute_scaling_saxs": 1.0,
    "absolute_scaling_waxs": 1.0,
    "thickness": 1.0,
    "normalization": "transmission",
    "save": False,
    "format": "dat",
}


class ForMAXConfigDialog(QDialog):
    def __init__(self, parent=None, initial: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ForMAX configuration")
        self._result = None
        self._fields: dict[str, object] = {}

        data = DEFAULT_FORMAX.copy()
        if initial:
            data.update(initial)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._fields["empty_i0"] = QLineEdit(str(data.get("empty_i0", "")))
        self._fields["empty_it"] = QLineEdit(str(data.get("empty_it", "")))
        self._fields["qSumStart"] = QLineEdit(str(data.get("qSumStart", "")))
        self._fields["qSumEnd"] = QLineEdit(str(data.get("qSumEnd", "")))
        self._fields["absolute_scaling_saxs"] = QLineEdit(str(data.get("absolute_scaling_saxs", "")))
        self._fields["absolute_scaling_waxs"] = QLineEdit(str(data.get("absolute_scaling_waxs", "")))
        self._fields["thickness"] = QLineEdit(str(data.get("thickness", "")))
        self._fields["normalization"] = QComboBox()
        self._fields["normalization"].addItems(["transmission"])
        self._fields["normalization"].setCurrentText(str(data.get("normalization", "transmission")))
        self._fields["save"] = QCheckBox()
        self._fields["save"].setChecked(bool(data.get("save", False)))
        self._fields["format"] = QLineEdit(str(data.get("format", "dat")))

        form.addRow("empty_i0", self._fields["empty_i0"])
        form.addRow("empty_it", self._fields["empty_it"])
        form.addRow("qSumStart", self._fields["qSumStart"])
        form.addRow("qSumEnd", self._fields["qSumEnd"])
        form.addRow("absolute_scaling_saxs", self._fields["absolute_scaling_saxs"])
        form.addRow("absolute_scaling_waxs", self._fields["absolute_scaling_waxs"])
        form.addRow("thickness", self._fields["thickness"])
        form.addRow("normalization", self._fields["normalization"])
        form.addRow("save", self._fields["save"])
        form.addRow("format", self._fields["format"])

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btnLoad = QPushButton("Load")
        self.btnSave = QPushButton("Save as...")
        self.btnApply = QPushButton("Apply")
        btn_row.addWidget(self.btnLoad)
        btn_row.addWidget(self.btnSave)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btnApply)
        layout.addLayout(btn_row)

        self.btnLoad.clicked.connect(self._on_load)
        self.btnSave.clicked.connect(self._on_save)
        self.btnApply.clicked.connect(self._on_apply)

    def _collect(self) -> dict:
        def _float(val):
            try:
                return float(val)
            except Exception:
                return val
        return {
            "empty_i0": _float(self._fields["empty_i0"].text().strip()),
            "empty_it": _float(self._fields["empty_it"].text().strip()),
            "qSumStart": _float(self._fields["qSumStart"].text().strip()),
            "qSumEnd": _float(self._fields["qSumEnd"].text().strip()),
            "absolute_scaling_saxs": _float(self._fields["absolute_scaling_saxs"].text().strip()),
            "absolute_scaling_waxs": _float(self._fields["absolute_scaling_waxs"].text().strip()),
            "thickness": _float(self._fields["thickness"].text().strip()),
            "normalization": self._fields["normalization"].currentText(),
            "save": self._fields["save"].isChecked(),
            "format": self._fields["format"].text().strip() or "dat",
        }

    def _on_load(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(CONFIG_DIR), "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, widget in self._fields.items():
            if key not in data:
                continue
            val = data[key]
            if isinstance(widget, QLineEdit):
                widget.setText(str(val))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(val))

    def _on_save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(CONFIG_DIR / "formax.json"), "JSON (*.json)")
        if not path:
            return
        data = self._collect()
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _on_apply(self):
        self._result = self._collect()
        self.accept()

    @staticmethod
    def get_config(parent=None, initial: Optional[dict] = None) -> Optional[dict]:
        dlg = ForMAXConfigDialog(parent, initial)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._result
        return None
