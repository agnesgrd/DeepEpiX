# 📘 Tuto 1 : Load & Preprocess Data

## 1️⃣ Go to Home Page

Open the app as explained in the **Setup & Installation** section. You should arrive at the 🏠 **Home Page**.  
If not, use the **☰ menu** at the top left to navigate there.

## 2️⃣ Choose a Subject

Use the **🔽 dropdown** menu to select a subject.  
👉 *Note:* The **📂 Open Folder** button only works if you have installed DeepEpiX locally.

You can open the following types of files:

- 📁 `.ds` folders
- 📄 `.fif` files
- 🧠 `4D` folders (must include at least: `rfDC-EEG`, `config` and `hs-file`)

## 3️⃣ Load and Access Metadata

When you click on **📥 Load**, the previous database memory will be cleared 🧹.

After loading, **⚙️ preprocessing parameters** become accessible.  
You can adjust these settings while exploring:

- 📊 Metadata (`raw.info`)
- 🗂️ Past annotations
- 📉 Power spectrum decomposition (as a function of frequency parameters)

## 4️⃣ Preprocess and Visualize

Clicking **⚡ Preprocess** will:

- 🧹 Filter and resample the data
- 💾 Store it in memory

This will take you to the **📈 Raw Visualization** main page.

To view metadata again, return to the **🏠 Home Page** and check the **📚 Database** table.
