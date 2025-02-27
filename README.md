<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://github.com/user-attachments/assets/bbe75d9c-204c-4890-8a9b-f8d5131b0032" alt="Logo" width="200" />
  </a>
</div>


<h3 align="center">DeepEpiX</h3>

  <p align="center">
    Software for annotation and automatic spike detection in MEG recordings
    <br />
    <a href="https://github.com/agnes_grd/DeepEpiX"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agnesgrd/DeepEpiX">View Demo</a>
    &middot;
    <a href="https://github.com/agnesgrd/DeepEpiX/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/agnesgrd/DeepEpiX/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
![Screenshot from 2025-01-10 11-12-11](https://github.com/user-attachments/assets/c27b6abe-b7ee-4860-8e35-27428fc1f34d)
![View_screenshot](https://github.com/user-attachments/assets/e15ef509-19cf-487a-a3e3-4f9d6f6a29b4)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python-shield]][Python-url]
* [![Dash][Dash-shield]][Dash-url]
* [![Plotly][Plotly-shield]][Plotly-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

??

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/agnesgrd/DeepEpiX.git
   ```
2. Go into DeepEpiX directory adn install the virtual environment for running a dash app :
   ```sh
   python3 -m venv .dashenv
   source .dashenv/bin/activate
   python3 -m pip install -r requirements-python3.9.txt
   ```
Alternatively, you can just install manually the core package listed in the requirements-python3.9.in file.
DeepEpiX was developed using Python 3.9, so this is what we recommend to use.

3. If you want to use prediction models, install the following virtual environment :
   #### TensorFlow
   ```sh
   python3 -m venv .tfenv
   source .tfenv/bin/activate
   python3 -m pip install -r requirements-tfenv.txt
   ```
  #### PyTorch
   ```sh
   python3 -m venv .torchenv
   source .torchenv/bin/activate
   python3 -m pip install -r requirements-torchenv.txt
   ```
   When running a prediction model, these environment would be call with their name .tfenv and .torchenv, they are expected to be in DeepEpiX folder.

4. Activate you dash environment and start running DeepEpiX in your terminal with
   ```sh
   . .dashenv/bin/activate
   python3 run.py
   ```
   then open the app in your web-browser under http://127.0.0.1:8050/
5. You're ready to use DeepEpiX Software

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
- [ ] Home Page
  - [ ] Visualize folder navigator (instead of entering full folder path)
- [ ] View Page
  - [ ] Plot topomap
  - [ ] Enable spike annotation
- [ ] Predict Page
  - [ ] Run a prediction model

See the [open issues](https://github.com/agnesgrd/DeepEpiX/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/agnesgrd/DeepEpiX](https://github.com/agnesgrd/DeepEpiX)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

[Python-url]: https://www.python.org
[Python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Dash-url]: https://dash.plotly.com
[Dash-shield]: https://img.shields.io/badge/Dash-red?style=for-the-badge&logo=Dash
[Plotly-url]: https://plotly.com/python
[Plotly-shield]: https://img.shields.io/badge/Plotly-black?style=for-the-badge&logo=Plotly
