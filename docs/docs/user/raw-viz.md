# Tuto 3: Visualize Preprocessed Signal

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">


## 1️⃣ Go to the Menu
If the left sidebar is collapsed, click the <i class="bi bi-layout-sidebar-inset"></i> **Open Sidebar** button. The sidebar contains 4 tabs:

- <i class="bi bi-hand-index-thumb"></i> **Select**
- <i class="bi bi-activity"></i> **Analyze**
- <i class="bi bi-stars"></i> **Spike Prediction**
- <i class="bi bi-floppy"></i> **Save**

## 2️⃣ Set Your Display Preferences
In the <i class="bi bi-hand-index-thumb"></i> Select tab:

- Choose a **montage** (channel layout), or if "channel selection" is active, pick specific **sensor groups** to view.
- Enable **annotations** to display on the main graph and in the annotation overview (below the time slider).
- Set **amplitude** (1–10) to adjust signal scaling (affects how compressed or expanded the signal appears vertically).
- Pick a c**olor** scheme (e.g., rainbow applies group-based coloring).
- Click the 🔄 **Refresh** button on the top-left **Modebar** to apply changes.

## 3️⃣ Once the Graph is Displayed
**Left Modebar:**

- 🔄 **Refresh** the graph after changing channels, amplitude, or color settings.
- ⏩ **Navigate between pages** (the signal is displayed in 2-minute chunks for performance; this duration can be modified in config.py).
- 🧭 **Jump to previous/next event** beyond the current view (default: all selected events; can be filtered by event type).

**Right Modebar:**

- 📸 Take a **snapshot** of the current view.
- 🔍 **Zoom in** on time and channels.
- 🖱️ **Pan horizontally** by clicking and dragging.
- ⏱️ **Zoom in/out** on the time axis.
- 🧼 **Autoscale** or **reset** to display the full signal duration.

**Time Range Slider:**

- Navigate through the signal timeline.
- Adjust the visible time range.

**Channel Slider:**

- Use your mouse or trackpad to scroll through channels vertically.

**📝 Annotation Overview**

- View annotation positions below the graph for quick reference.
