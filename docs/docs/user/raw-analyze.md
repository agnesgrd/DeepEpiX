# Tuto 4: Analyze Preprocessed Signal

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">


## 1️⃣ Go to the Menu
If the left sidebar is collapsed, click the <i class="bi bi-layout-sidebar-inset"></i> **Open Sidebar** button. The sidebar contains 4 tabs:

- <i class="bi bi-hand-index-thumb"></i> **Select**
- <i class="bi bi-activity"></i> **Analyze**
- <i class="bi bi-stars"></i> **Spike Prediction**
- <i class="bi bi-floppy"></i> **Save**

## 2️⃣ Analyze Your Signal Manually
In the <i class="bi bi-activity"></i> Analyze tab:

- **Activate the Topomap**: click once on the button to activate the feature (click again to deactivate). Then click on the signal at the desired timepoint to view spatial maps.
- **Add annotations**: click or select a segment on the graph to auto-fill onset and duration (set duration = 0 for punctual events). Add a label and click **Add New Event**. It will appear in the annotation options (make sure it is selected to be visible on the graph).
- **Delete annotations**: select a segment and click **Deleted Selected Event(s)** to remove the current event(s) in the selection.
- **To delete an entire event category**, go to the **Select** tab, choose the event type, and click **Delete**, then confirm.

## 3️⃣ Run a Prediction Model
In the <i class="bi bi-stars"></i> **Spike Prediction** tab:

- **Select your model**.
- **Optional**: enable **sensitivity analysis** and **onset adjustement**.
    - Sensitivity analysis (SmoothGrad) averages gradients over noisy inputs to highlight the most influential regions of the signal (available for simple TensorFlow models).
    - You can also choose GFP-based alignment, which adjusts the spike onset to the peak of Global Field Power for greater accuracy.
- Click **Run Prediction**. It should complete in under a minute (on GPU).
- When results appear:
    - **Adjust the threshold** to refine spike detection based on the displayed probability distribution.
    - **Rename the result** (default: model_name_threshold_value) and click **Save** to make it selectable in the graph view.

> If SmoothGrad was selected, a new color layer will be added in the **Select** panel. It highlights predicted spike regions on the signal.

> Running the same model again for the same subject will return cached results instantly—you can re-save with a different threshold.