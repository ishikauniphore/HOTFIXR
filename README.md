# Setting up the environment
Please refer to `setup_verl_env.sh`. You can most likely give it to Claude Code or any other AI you like to use and ask it to set up the environment for you with the directions in the file. If there is any issue with CUDA, then please refer to CUDA_802_FIX.md (it's an issue that I frequently faced with Duplo Cloud servers, and had to find a permanent fix for -- hopefully it solves any issues you also have).

# Before implementing
Please make sure you are logged into HuggingFace. WandB is optional.

# Running HOTFIXR via bash script
Please refer to `run_HOTFIXR.sh`. It has the detailed commands laid out, with all the variables provided so you can customize whatever you'd like.

# Running HOTFIXR via Python
For a more cleaner abstraction, you can also use the HOTFIXR API as shown in `run_HOTFIXR.py`. Note, this file assumed 8 NVIDIA A100's (equivalent to a p4d.24xlarge on Duplo Cloud), but can be easily customized.

After cloning the repo, install the HOTFIXR package by:

```
cd /home/ubuntu/HOTFIXR && pip install -e .
```

NOTE: This is a bit of an unclean implementation, whenever a path or filename is required, please provide the full path of where to store that path/file.

# Datasets
We've upload all the data on SharePoint ([Data Zip](https://uniphore-my.sharepoint.com/:u:/p/ishika_agarwal/IQA6a2HzoGC4SI6Wi1kyEwaZAQpyvAU50E5jrr3-5WZTRSQ?e=zPvUBo)) we used for our results. You are more then welcome to use those. We've also provided the code for processing the datasets for VERL/GRPO in `data/preprocess_nemotron.py`.

# Push notifications when your experiment is finished
GRPO experiments can take a long time! Instead of waiting to find out when it is finished, you can send push notifications to your phone. Download the ntfy app ([https://ntfy.sh/](https://ntfy.sh/)), and add this to your `.bashrc` file (make sure to input your own unique code):

```
notify() {
  curl -s -d "${1:-done}" ntfy.sh/[YOUR UNIQUE CODE HERE]
}
```

Next, put this command at the end of your bash scripts:

```
notify "experiment finished"
```