# Environment Fix Runbook — move off Microsoft Store Python

Goal: rebuild the lerobot `.venv` on an official python.org 3.12 instead of the
Microsoft Store build, without losing the CUDA torch stack or hidapi setup.

Work top to bottom. Steps 0–3 are prep/safety. Step 4 is the destructive one.

---

## Phase A — Capture current state (safety net)

- [ ] **0. Record what's installed (so nothing is guessed on reinstall).**
  ```powershell
  cd c:\Users\aa24\PhD\lerobot
  .\.venv\Scripts\python.exe -m pip freeze > frozen-before-rebuild.txt
  ```
  Keep `frozen-before-rebuild.txt`. It is your rollback reference.

- [ ] **1. Note the exact torch build you must preserve.**
  Current: `torch 2.7.0+cu128`, CUDA 12.8, GPU working.
  You will reinstall this specific wheel in Step 6 — NOT a plain `pip install torch`.

- [ ] **2. Locate your hidapi.dll.**
  It currently lives in `.venv\Scripts\hidapi.dll` (needed for SpaceMouse/meca500).
  Copy it somewhere safe so you can restore it after the rebuild:
  ```powershell
  Copy-Item .\.venv\Scripts\hidapi.dll $env:USERPROFILE\Desktop\hidapi.dll
  ```

---

## Phase B — Install a real Python and stop the Store from interfering

- [ ] **3. Install python.org 3.12.**
  ```powershell
  winget install Python.Python.3.12
  ```
  Lands in `C:\Users\aa24\AppData\Local\Programs\Python\Python312\` (writable, normal layout).
  Verify:
  ```powershell
  & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
  ```

- [ ] **4. Disable the Store app-execution aliases.**
  Settings → Apps → Advanced app settings → App execution aliases →
  toggle **OFF**: `python.exe` and `python3.exe`.
  (Stops the zero-byte WindowsApps stubs from hijacking `python`.)

---

## Phase C — Rebuild the venv (DESTRUCTIVE step)

- [ ] **5. Delete and recreate the venv on the new interpreter.**
  ```powershell
  cd c:\Users\aa24\PhD\lerobot
  Remove-Item -Recurse -Force .venv
  & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  ```
  Confirm the base is no longer the Store build:
  ```powershell
  Get-Content .\.venv\pyvenv.cfg    # 'home' should point at ...\Programs\Python\Python312
  ```

- [ ] **6. Reinstall CUDA torch FIRST (before other deps pull a CPU torch).**
  ```powershell
  pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  # expect: 2.7.0+cu128 True
  ```

- [ ] **7. Reinstall lerobot + your extras.**
  ```powershell
  pip install -e .
  ```
  Then any meca500/camera extras you normally install. If unsure, diff against
  `frozen-before-rebuild.txt` and install anything missing.

- [ ] **8. Restore hidapi.dll into the new venv.**
  ```powershell
  Copy-Item $env:USERPROFILE\Desktop\hidapi.dll .\.venv\Scripts\hidapi.dll
  ```

---

## Phase D — Editor + git hygiene

- [ ] **9. Pin the VS Code interpreter** (in `.vscode/settings.json`):
  ```json
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
  ```
  Then in VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick the `.venv` one.

- [ ] **10. (Optional) Set git line-ending behavior.**
  ```powershell
  git config core.autocrlf true
  ```

---

## Phase E — Verify, then clean up

- [ ] **11. Smoke test the rebuilt environment.**
  ```powershell
  python -c "import sys; print(sys.executable)"     # -> ...\.venv\Scripts\python.exe
  python -c "import torch; print(torch.cuda.is_available())"   # -> True
  ```
  Run a real meca500 / camera / SpaceMouse script to confirm hidapi + hardware work.

- [ ] **12. (Optional, ONLY after everything works) Remove Store Pythons.**
  ```powershell
  Get-AppxPackage *PythonSoftwareFoundation.Python* | Remove-AppxPackage
  ```
  Leave your separate `leisaac_envhub` conda env untouched — it is unrelated.

- [ ] **13. Delete this runbook + scratch files** once done:
  `ENV_FIX_RUNBOOK.md`, `frozen-before-rebuild.txt`, Desktop `hidapi.dll`.

---

### Notes
- requires-python for lerobot is `>=3.12`; we choose 3.12 (not 3.13/3.14) for the
  most reliable Windows wheel coverage, especially PyTorch.
- Your `leisaac_envhub` conda env is a separate environment and is NOT affected by
  any of this.
