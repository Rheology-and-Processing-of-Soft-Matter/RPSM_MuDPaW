import csv
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class EmbedListWindow(tk.Toplevel):
    def __init__(self, master, rows, on_save):
        super().__init__(master)
        self.title("Embed list")
        self.minsize(720, 420)
        self.rows = rows
        self.on_save = on_save

        self.columns = self._compute_columns(rows)
        self.tree = ttk.Treeview(self, columns=self.columns, show="headings", selectmode="browse")
        self._configure_columns()
        self.tree.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=8, pady=8)
        # Allow the tree to stretch; buttons remain minimal
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)

        ttk.Button(self, text="Add", command=self._add).grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(self, text="Edit", command=self._edit).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(self, text="Delete", command=self._delete).grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(self, text="Save to file", command=self._save_file).grid(row=1, column=3, padx=4, pady=4, sticky="ew")

        # inline edit support
        self.tree.bind("<Double-1>", self._begin_inline_edit)
        self._edit_entry = None
        self._edit_item = None
        self._edit_column = None

        self.refresh()

    def _compute_columns(self, rows):
        base = ["Name", "Detail", "Scan interval", "background", "bg_scale"]
        extra = set()
        for row in rows:
            for k in row.keys():
                if k is None:
                    continue
                key_simple = re.sub(r"[\\s_-]+", "", str(k).strip().lower())
                if key_simple.startswith("qrange"):
                    extra.add(k)
        if not extra:
            extra = {"q_range1"}

        def _q_index(name: str) -> int:
            m = re.search(r"\d+", name)
            return int(m.group()) if m else 0

        ordered_extra = sorted(extra, key=_q_index)
        return base + ordered_extra

    def _configure_columns(self):
        widths = {
            "Name": 100,
            "Detail": 240,
            "Scan interval": 160,
            "background": 100,
        }
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths.get(col, 100), anchor="w")

    def refresh(self):
        # Recompute columns if new ranges were added
        new_cols = self._compute_columns(self.rows)
        if new_cols != self.columns:
            self.columns = new_cols
            self.tree["columns"] = self.columns
            self._configure_columns()
        self.tree.delete(*self.tree.get_children())
        for row in self.rows:
            values = [row.get(col, "") for col in self.columns]
            self.tree.insert("", "end", values=values)
        self._cancel_inline_edit()

    def _get_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.index(sel[0])

    def _add(self):
        row = self._prompt_row()
        if row:
            self.rows.append(row)
            self.refresh()
            self.on_save(self.rows)

    def _edit(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showerror("Edit", "Select a row to edit.", parent=self)
            return
        row = self.rows[idx]
        new_row = self._prompt_row(row)
        if new_row:
            self.rows[idx] = new_row
            self.refresh()
            self.on_save(self.rows)

    def _delete(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showerror("Delete", "Select a row to delete.", parent=self)
            return
        del self.rows[idx]
        self.refresh()
        self.on_save(self.rows)

    # ------------- inline editing -------------
    def _begin_inline_edit(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        if not item or not column_id:
            return
        col_index = int(column_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.columns):
            return
        self._cancel_inline_edit()
        x, y, width, height = self.tree.bbox(item, column_id)
        value = self.tree.set(item, self.columns[col_index])
        entry = ttk.Entry(self.tree)
        entry.insert(0, value)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        entry.bind("<Return>", lambda _e: self._commit_inline_edit(entry.get()))
        entry.bind("<FocusOut>", lambda _e: self._commit_inline_edit(entry.get()))
        self._edit_entry = entry
        self._edit_item = item
        self._edit_column = self.columns[col_index]

    def _commit_inline_edit(self, new_value: str):
        if not self._edit_entry or not self._edit_item or not self._edit_column:
            return
        try:
            idx = self.tree.index(self._edit_item)
            if 0 <= idx < len(self.rows):
                self.rows[idx][self._edit_column] = new_value
                self.refresh()
                self.on_save(self.rows)
        finally:
            self._cancel_inline_edit()

    def _cancel_inline_edit(self):
        if self._edit_entry:
            try:
                self._edit_entry.destroy()
            except Exception:
                pass
        self._edit_entry = None
        self._edit_item = None
        self._edit_column = None

    def _save_file(self):
        path = filedialog.asksaveasfilename(
            title="Save embed list CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.columns)
                writer.writeheader()
                for row in self.rows:
                    writer.writerow({col: row.get(col, "") for col in self.columns})
            messagebox.showinfo("Save", f"Saved {len(self.rows)} entries to {path}", parent=self)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save", f"Failed to save list:\n{exc}", parent=self)

    def _prompt_row(self, initial=None):
        dlg = tk.Toplevel(self)
        dlg.title("Row")
        dlg.transient(self)
        dlg.grab_set()
        vals = {col: "" for col in self.columns}
        if initial:
            vals.update(initial)
        entries = {}
        for i, key in enumerate(self.columns):
            ttk.Label(dlg, text=key).grid(row=i, column=0, padx=6, pady=4, sticky="w")
            var = tk.StringVar(value=vals.get(key, ""))
            ent = ttk.Entry(dlg, textvariable=var, width=50)
            ent.grid(row=i, column=1, padx=6, pady=4, sticky="ew")
            entries[key] = var
        dlg.columnconfigure(1, weight=1)
        result = {}

        def _ok():
            for k, var in entries.items():
                result[k] = var.get().strip()
            dlg.destroy()

        ttk.Button(dlg, text="OK", command=_ok).grid(row=len(self.columns), column=0, columnspan=2, pady=6)
        dlg.wait_window()
        return result if result else None
