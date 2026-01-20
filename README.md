**Getting Started**
**Step 1**: Clone the Repository
git clone https://github.com/ptree789/MUEUDA

**Step 2**: Prepare the Datasets
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

**Step 3**: Configure Training Script

Edit the script located at:
scripts/cocoop/LCMUDA_train.sh

Replace the cd path and the DATA variable with your own local paths. Set DATASET to one of the following options:officehome, of31, domainnet
Each option corresponds to the OfficeHome, Office31, and DomainNet datasets respectively.

**Step 4**: Set Source and Target Domains
Modify the --source-domains and --target-domains arguments in LCMUDA_train.sh to reproduce different settings:
UniSDA (Universal Single-source Domain Adaptation)
Set --source-domains to one specific domain.

UniMDA (Universal Multi-source Domain Adaptation)
Set --source-domains to multiple domains.
--target-domains should always be set to the name of the target domain.

Step 5: Run Training
bash scripts/cocoop/LCMUDA_train.sh

For example, if:
DATASET=officehome
--source-domains=art clipart product
--target-domains=real_world
the script will reproduce the UniMDA 2R results on the OfficeHome dataset.
