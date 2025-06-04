```
DeepEpiX/

├── data/                 # Put your data here - when built with Docker, local data directory is mounted on it.
│   ├── patient_1.ds      
│   ├── patient_2.fif
│   ├── patient_3_4D/
│   │   ├── rfDC_EEG
│   │   ├── config
│   │   └── hs_file

├── docs/                 # Use mkdocs

├── requirements/         # Use pip-tools to generate .txt from .in

├── src/                  
│   ├── assets/           # Static image/logo/icons
│   ├── cache-directory/  # Cached intermediate data or results - cleaned every time a new subect is loaded
│   ├── callbacks/        
│   │   ├── utils/
│   │   │   ├── viz_utils.py
│   │   └── viz.py        # Chainable functions that are automatically called whenever a UI element on viz.py page is changed
│   ├── layout/           # UI elements definition
│   ├── model_pipeline/   # Extracted from https://github.com/pmouches/DeepEpi/tree/main/pipeline with some modifications
│   ├── models/           # ML models from from https://github.com/pmouches/DeepEpi/tree/main/
│   ├── pages/            # Multi-page app
│   │   ├── home.py 
│   │   ├── viz.py 
│   │   ├── ...
│   │   └── settings.py
│   ├── static/           # Static files
│   ├── config.py         # Configuration settings and constants
│   └── run.py            # Entry point to run the multi-page app


├── DeepEpiX.def          # Singularity definition file for containerization
├── Dockerfile            # Docker definition file for containerization
└── README.md 
```

This structure is constrained by the multi-page construction app. Each page.py is placed in pages/. Each page has its callbacks repartis in callbacks/ and its layout.