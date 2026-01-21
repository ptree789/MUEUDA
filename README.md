## **Getting Started**

## **Step 1**: Clone the Repository

git clone https://github.com/ptree789/MUEUDA

## **Step 2**: How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch]. so you need to install the `dassl` environment first. 

1) cd Dassl.pytorch/

# Create a conda environment
2) conda create -y -n dassl python=3.8

# Activate the environment
3) conda activate dassl

4) # Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop

# After that, run `pip install -r requirements.txt` under `MUEUDA/` to install a few more packages required by [CLIP] (this should be done when `dassl` is activated)
5) cd ..
   pip install -r requirements.txt

## **Step 3**: Prepare the Datasets

Create a folder named DATA in the project root. Download the OfficeHome, Office31, and DomainNet datasets and place them under the directory.
Download the datasets from the sources:

**OfficeHome**:

https://www.hemanthdv.org/officeHomeDataset.html

**Office31**:

https://www.kaggle.com/datasets/mei1963/office31

**DomainNet (only Painting, Real, and Sketch domains are used)**:

https://ai.bu.edu/M3SDA/#dataset

The directory structure should look like:

data/

 ├── officehome/
 
 ├── office31/
 
 └── domainnet/

## **Step 4**: Configure Training Script

Edit the script located at:
scripts/cocoop/LCMUDA_train.sh

Replace the cd path and the DATA variable with your own local paths. Set DATASET to one of the following options:officehome, of31, domainnet

Each option corresponds to the OfficeHome, Office31, and DomainNet datasets respectively.

## **Step 5**: Set Source and Target Domains
Modify the --source-domains and --target-domains arguments in LCMUDA_train.sh to reproduce different settings:

UniSDA (Universal Single-source Domain Adaptation)

Set --source-domains to one specific domain.

UniMDA (Universal Multi-source Domain Adaptation)

Set --source-domains to multiple domains.

--target-domains should always be set to the name of the target domain.

## **Step 6**: Run Training:

bash scripts/cocoop/LCMUDA_train.sh

For example, if:

DATASET=officehome

--source-domains=art clipart product

--target-domains=real_world

the script will reproduce the UniMDA 2R results on the OfficeHome dataset.
