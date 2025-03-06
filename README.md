<div align="left">
    <img src="media/tistools.webp" width="40%" align="left" style="margin-right: 15px"/>
    <div style="display: inline-block;">
        <div id="user-content-toc">
          <ul style="display: inline-block; vertical-align: middle; margin-top: 0;border-bottom: none;list-style: none;">
            <summary>
              <h1>tistools</h1>
            </summary>
          </ul>
        </div>
        <p>
    <em><code>❯ Tools to analyse the output of TIS or RETIS simulations</code></em>
</p>
        <p>
    <img src="https://img.shields.io/github/license/annekegh/tistools?style=flat-square&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
    <img src="https://img.shields.io/github/last-commit/annekegh/tistools?style=flat-square&logo=git&logoColor=white&color=0080ff" alt="last-commit">
    <img src="https://img.shields.io/github/languages/top/annekegh/tistools?style=flat-square&color=0080ff" alt="repo-top-language">
    <img src="https://img.shields.io/github/languages/count/annekegh/tistools?style=flat-square&color=0080ff" alt="repo-language-count">
</p>
        <p>Built with the tools and technologies:</p>
        <p>
    <img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
    <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
</p>
    </div>
</div>
<br clear="left"/>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)<!-- - [Contributing](#contributing) -->
- [License](#license)
- [Authors](#authors)

---

## Overview

`TISTOOLS` is a Python program that extracts the information from the output of TIS or RETIS simulations performed with the program PyRETIS (see [www.pyretis.org](http://www.pyretis.org)) or programs with the PyRETIS output file format. It can be used to assess the statistics of the path ensembles and those of the order parameter(s).

---

## Features

- Extract information from TIS or RETIS simulations
- Plot distributions of order parameters in each path ensemble
- Analyze statistics of path ensembles
- Examine order parameter statistics
- Support for various trajectory cleaning and analysis tools

---

## Project Structure

```sh
└── tistools/
    ├── LICENSE
    ├── MANIFEST.in
    ├── README.md
    ├── examples
    │   ├── hist.oxygen.txt
    │   ├── recalcjob.sh
    │   ├── ringdistjob.sh
    │   └── wholemakejob.sh
    ├── lib
    │   ├── __init__.py
    │   ├── analyze_op.py
    │   ├── block_error_analysis.py
    │   ├── cleaner.py
    │   ├── distrib_op.py
    │   ├── free_energy.py
    │   ├── pathlengths.py
    │   ├── pathproperties.py
    │   ├── reading.py
    │   ├── repptis_analysis.py
    │   ├── repptis_msm.py
    │   ├── repptis_pathlengths.py
    │   └── writing.py
    ├── pyproject.toml
    ├── scripts
    │   ├── subsampler_chunk
    │   ├── tistools-clean-trajectories
    │   ├── tistools-cleaner
    │   ├── tistools-distr-op
    │   ├── tistools-distr-paths
    │   ├── tistools-distrib-path
    │   ├── tistools-more-distr
    │   └── tistools-remove-waters
    ├── setup.py
    └── test
        ├── analysis_notebook.ipynb
        ├── block_error_analysis.ipynb
        ├── msm_pptis.ipynb
        ├── msm_pptis_clean.ipynb
        └── msm_pptis_mfpt.ipynb
```

---

## Getting Started

### Prerequisites

Before getting started with tistools, ensure your runtime environment meets the following requirements:

- **Operating System:** Linux or MacOS
- **Programming Language:** Python >= 3
- **Third-party Libraries:** 
  - NumPy
  - Matplotlib

### Installation

Install tistools using one of the following methods:

**Build from source:**

1. Clone the tistools repository:
```sh
❯ git clone https://github.com/annekegh/tistools.git
```

2. Navigate to the project directory:
```sh
❯ cd tistools
```

3. Install the project dependencies:
```sh
❯ pip install .
```

or for development:
```sh
❯ pip install -e .  # code is editable
```

If you want to ensure pip installs in the correct environment:
```sh
❯ which python  # check that this is the python you want to use
❯ python -m pip install -e .
```

### Usage

Generate a histogram of the order parameter with tistools using the following command:
```sh
❯ tistools-distr-op
```

For example, to plot the distributions of the order parameter in each path ensemble:
```sh
❯ tistools-distr-op -i myproject/system8/
```

Use the help function to see the other optional arguments:
```sh
❯ tistools-distr-op -h
```

---
<!-- 
## Contributing

- **💬 [Join the Discussions](https://github.com/annekegh/tistools/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/annekegh/tistools/issues)**: Submit bugs found or log feature requests for the `tistools` project.
- **💡 [Submit Pull Requests](https://github.com/annekegh/tistools/pulls)**: Review open PRs, and submit your own PRs.

--- -->

## License

See [LICENSE](LICENSE) file.

---

## Authors

- An Ghysels, BioMMedA, Ghent University
- Wouter Vervulst, BioMMedA, Ghent University
- Elias Wils, BioMMedA, Ghent University
- Sina Safaei, BioMMedA, Ghent University
