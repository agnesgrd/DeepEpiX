# 🧠 MEG/EEG GUI Software

Welcome to the documentation of DeepEpiX, a Dash-based MEG/EEG GUI Software.  
This app provides an interactive web interface for loading, preprocessing, annotating and analyzing raw MEG/EEG files, and last but not least: running prediction models.
<!-- 
![My Photo](/images/preprocessing.png) -->
![My Photo](images/viz.png)

---

## 🚀 What This App Does

- ✅ Load raw MEG/EEG datasets (`.ds`, `.fif`, or 4D-compatible)
- ✅ Set frequency filtering parameters (resampling, high-pass, low-pass, notch)
- ✅ Detect ECG peaks via channel hinting
- ✅ Drop bad channels
- ✅ Visualize event statistics, power spectral density, topomap, ICA...
- ✅ Display temporal signal with various options
- ✅ Create custom sensors layout (montage)
- ✅ Run prediction models
- ✅ Measure their performances

## 🤔 What This App Should Do in the Future

- 💡 Allow continuous learning of prediction models

---

## 🗂 App Structure

The app is structured around **pages**, **layout** and **callbacks**.


## 📖 Docs Navigation

- [📦 Folder Structure](structure.md)
- [🧩 Page: Preprocessing](pages/preprocessing.md)
- [📊 Page: PSD](pages/psd.md)
- [🔄 Callback Glossary](callbacks/index.md)
- [👨‍💻 Developer Setup](dev/setup.md)

---

## 👩‍💻 Who Is This For?

- Developers extending or maintaining the app
- Researchers and clinicians using the app for MEG/EEG studies
- Contributors improving UI, performance, or adding features