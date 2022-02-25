# Assignment 2: Multi-Scale Pyramids

# Instructions 
1. Follow [Getting Started](https://github.com/wis-intro-vision/getting-started) to setup your environment.
2. Clone your repository (`ex2-pyramids-<yourname>`).
3. Open the notebook, by running the provided start script `scripts/run-notebook.bat` (similar script exists for Mac/Linux users). This script activates the `wis-cv` conda environment, and open the assignment's notebook.
4. Follow the instructions in the notebook.  
**Note:** the included papers can be found in the `papers/` subdirectory. The original images can be found in `data/`.
5. Once finished (or more frequently), commit your changes and push them to GitHub. Make sure you push them to `origin/master` (if you don't know what that means, you'll probably be fine).

## Useful Resources
1. If you're new to `git`, you may want to take a look at the following tutorials:
    - [git - the simple guide](http://rogerdudler.github.io/git-guide/)
    - [Git-it](https://github.com/jlord/git-it-electron#what-to-install): a hands-on tutorial.
2. If you're new to Python, we recommend you read [this tutorial](http://cs231n.github.io/python-numpy-tutorial/).
3. If you're unfamiliar with Jupyter notebooks, you may find [this tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) useful.

# Troubleshooting
## Error when running the `run-notebook` script
### Windows
1. If you get the following error message:
    > 'conda' is not recognized as an internal or external command, operable program or batch file.

    Then open *Anaconda Prompt* (from *Start* menu), drag the script into it, and press *Enter* to run it.

### Linux and Mac
1. If you get the following error message:
    > bash: conda: command not found

    Then open *bash* and run the following command (to add *Anaconda3* to your path):
    ```bash
    echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    ```
    If Anaconda is installed at a different location, change the script accordingly. This will work for Anaconda2 and Miniconda with appropriate changes.