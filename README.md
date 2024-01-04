# llama2

to watch and apply : https://www.youtube.com/watch?v=bmpjT0T4IDY

## Access to the file
To access this file and play with it, you need to have git and conda install. 

### Git
You need git in order to download this repository. In your terminal (cmd) you must enter:

```bash
git clone github.com/teilomillet/llama2 
# OR 
git clone git@github.com:teilomillet/llama2

# THEN
cd llama2
```

After downloading the repository you will need to go inside it

### Conda
You need `conda` installed. [miniconda here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)


### Create the conda environment
Create and activate conda environment:

```bash
conda env create -f environment.yaml
conda activate mojo-llama
```

### Auto set mojo environment
To automatically set Mojo to use the python environment when you activate it:

#### Macos/Linux
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
export MOJO_PYTHON_LIBRARY="$(find $CONDA_PREFIX/lib -iname 'libpython*.[s,d]*' | sort -r | head -n 1)"
echo "export MOJO_PYTHON_LIBRARY=\"$MOJO_PYTHON_LIBRARY\"" > $CONDA_PREFIX/etc/conda/activate.d/export-mojo.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset MOJO_PYTHON_LIBRARY" > $CONDA_PREFIX/etc/conda/deactivate.d/unset-mojo.sh
```

## Usage
Activate the environment and run the program:

```bash
conda activate mojo-llama
mojo main.mojo
```