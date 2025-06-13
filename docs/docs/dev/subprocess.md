# ⚙️ Running External Scripts with `subprocess` in Python

This section describes how to run a Python script in a **separate virtual environment** using the `subprocess` module. This approach is useful when models or pipelines require isolated environments like TensorFlow or PyTorch.

---

## 📁 Use Case: Executing a Model in a Separate Virtual Environment

To simplify development and reduce dependency conflicts, we **separated the Dash environment** from heavy ML frameworks like TensorFlow and PyTorch:

- The Dash app runs inside a lightweight environment:  
    - **`.dashenv`**
  
- The model listed in `models/` require dedicated ML environments:
    - **`.tfenv`**
    - **`.torchenv`**
  
When the user decides to run a model, the app should detect the required backend and delegate execution to the appropriate Python binary.

Below is an example script we want to run using a specific ML environment (`model_pipeline/run_model.py`).

```python
# model_pipeline/run.model.py
from ... import run_model_pipeline

if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    subject_folder_path = sys.argv[3]
    results_path = sys.argv[4]
    threshold = float(sys.argv[5])
    adjust_onset = sys.argv[6]
    bad_channels = sys.argv[7]

    run_model_pipeline(model_path, model_type, subject_folder_path, results_path, threshold, adjust_onset, bad_channels)
```

---

## ✅ Step-by-Step Example

1. **Classical Import**
    ```python
    # callbacks/predict_callbacks.py
    import subprocess
    import os
    import time
    from pathlib import Path
    ```
2. **Backend Detection**

    ```python
        # Select the Python executable based on the virtual environment
        if "TensorFlow" in venv:
            ACTIVATE_ENV = str(config.TENSORFLOW_ENV / "bin/python")      
        elif "PyTorch" in venv:
            ACTIVATE_ENV = str(config.TORCH_ENV / "bin/python")
    ```
    The variable `venv` is determined earlier in the function based on the model file's extension (`.pth`, .`keras`, `h5`).
    `TENSORFLOW_ENV` and `TORCH_ENV` are paths to the respective virtual environments, defined in the `config` module.

3. **Command Definition**

    The command should be passed as a list of strings, where each list element is a separate argument.
    ```python

        # Build the command to execute
        command = [
            ACTIVATE_ENV,
            "model_pipeline/run_model.py",         # Script to run
            str(model_path),                       # Argument 1
            str(venv),                             # Argument 2
            str(subject_folder_path),              # Argument 3
            str(cache_dir),                        # Argument 4
            str(threshold),                        # Argument 5
            str(adjust_onset),                     # Argument 6
            str(channel_store.get('bad', []))      # Argument 7
        ]
    ```

4. **Python Path**

    If the `PYTHONPATH` environment variable isn't set correctly when using `subprocess`, Python may not be able to locate local modules properly, leading to import failures.

    ```python

        # Set the working directory and environment variables
        working_dir = Path.cwd()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(working_dir)  # Ensure imports work correctly
    ```

5. **Subprocess execution**
    ```python
        # Run the subprocess
        try:
            subprocess.run(command, env=env, text=True)  # Run command in isolated process

        except Exception as e:
            print(f"⚠️ Error running model: {e}")
    ```
    Use ```text=True``` to ensure output is treated as text (not bytes).
    To suppress output, uncomment ```stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL```.


--- 
## 💡 Tips

- To retrieve model results afterward, make sure your script saves outputs (e.g., CSVs, logs) into the `cache-directory`/.

- Async option: use `multiprocessing` or `ThreadPoolExecutor` if you don’t want the Dash app to freeze during long model runs.


---

## 📘 When to Use This
Use a subprocess when:

- You want to isolate dependencies (e.g., TensorFlow vs. PyTorch).

- You need to run long or blocking processes separately.
