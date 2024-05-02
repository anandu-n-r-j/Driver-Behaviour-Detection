Detecting the behaviour of drivers while driving the vehicles...
Environment Setup

#### Installation

To use this project, follow these steps to set up your environment:

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Create Virtual Environment**: Navigate to the project directory and create a virtual environment.
   ```bash
   cd project-directory
   python3 -m venv venv
   ```

3. **Activate Virtual Environment**: Activate the virtual environment.
   ```bash
   source venv/bin/activate
   ```

4. **Install Dependencies**: Install required dependencies using the provided requirements file.
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Data Preparation

1. **Download CelebA Dataset**: Manually download the CelebA dataset from [this link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

2. **Data Storage**: Store the downloaded dataset in a directory of your choice.

3. **Preprocessing**: If preprocessing is required, refer to the preprocessing section in the provided Jupyter notebooks.

#### Running Jupyter Notebooks

1. **Start Jupyter Notebook**: Launch Jupyter Notebook.
   ```bash
   jupyter notebook
   ```

2. **Open Notebooks**: Navigate to the project directory and open the provided Jupyter notebooks (`main_code.ipynb` and `evaluation.ipynb`).

3. **Execute Cells**: Run the cells in the notebooks to execute the code.

#### Model Training and Evaluation

- Follow the instructions provided in the notebooks (`main_code.ipynb` for training and `evaluation.ipynb` for evaluation) to train the model, evaluate performance, and generate images.

#### Code directory structure

├── README.md          <- The top-level README for developers. 
├── data 
│   ├── input          <- Folder containing celebA raw data.
│   └── output         <- Folder to save .ckpts, generated images and plots.
│       ├── checkpoints     <- Weights of model saved as .h5 files
│       └── generated_images <- Folder containing generated images.
        └── figures          <- Folder containing plots of training progress.
├── docs               <- A folder for documentation 
├── notebooks 
│   ├── main_code.ipynb      <- The main code(containing the model).
│   └── evaluation.ipynb     <- The evaluation of the model.
├── requirements.txt   <- Required modules to be installed.  
├── references         <- Data dictionaries, manuals, and all other explanatory materials. 

#### Additional Notes

- Ensure that you have the necessary data stored in the appropriate directories as per the instructions in the notebooks.
- Adjust file paths and configurations as needed for your environment.
- Feel free to modify the notebooks to suit your requirements or add additional functionality.
- For any questions or issues, refer to the documentation or reach out to the project maintainers.

This README provides a basic guide for setting up the environment, running Jupyter notebooks, training the model, and evaluating performance. Customize the steps according to your specific use case and environment.
