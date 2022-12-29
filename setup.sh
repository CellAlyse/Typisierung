# ask if conda environment should be created
read -p "Do you want to create a conda environment? [y/n] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # create a conda environment namen cellalyse, the verison is 3.9
    conda create -n cellalyse python=3.9 -y

    # activate the environment
    conda activate cellalyse

    # install the packages
    pip install -r requirements.txt


fi

# ask if dataset should be downloaded
read -p "Do you want to download the dataset? [y/n] " -n 1 -r
echo
# if yes, download the dataset
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python download.py
fi

# if no, exit
if [[ $REPLY =~ ^[Nn]$ ]]
then
    exit 1
fi

