# Llama 2

Ce projet vise à reconstruire l'architecture de llama 2. J'ai implémenté les fonctionnalités suivantes :

- GQA (Grouped-Query Attention) [Lien vers le papier](https://arxiv.org/pdf/2305.13245.pdf)
- RoFormer (Rotary Position Embedding) [Lien vers le papier](https://arxiv.org/pdf/2104.09864.pdf)
- RMSNorm (Root Mean Square Layer Normalization) [Lien vers le papier](https://arxiv.org/pdf/1910.07467.pdf)
- SwigLU [Lien vers le papier](https://arxiv.org/pdf/2002.05202v1.pdf)
- KV cache [Lien vers le papier](https://arxiv.org/pdf/2211.05102.pdf)
- Top P [Lien vers le papier](https://arxiv.org/pdf/1904.09751.pdf)

Pour reproduire ou essayer ce projet, il est nécessaire de télécharger les checkpoints du modèle fournis par Meta. Dans notre cas, téléchargez le modèle 'llama-2-7b' depuis en suivant la démarche de [ce lien](https://github.com/facebookresearch/llama). 

L'objectif est de servir de point de départ pour des tests ou approfondir l'apprentissage et la compréhension de l'architecture des LLMs.

## Accès et Modification du Fichier
Pour accéder, télécharger et modifier ce fichier, vous devez avoir installé Git et Conda.

### Git
Pour cloner ce dépôt et accéder au code, ouvrez votre terminal et entrez les commandes suivantes :

```bash
git clone github.com/teilomillet/llama2 
# OU 
git clone git@github.com:teilomillet/llama2

# PUIS
cd llama2
```

### Conda
Conda doit être installé pour la suite du processus. Vous pouvez installer [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)


#### Création d'un Environnement Conda
Pour créer et activer un environnement Conda, entrez les commandes suivantes dans votre terminal :

```bash
conda env create -f environment.yaml
conda activate llama2
```

### Configuration Automatique de l'Environnement Mojo
Pour configurer automatiquement Mojo afin qu'il utilise l'environnement Conda activé :

#### Macos/Linux
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
export MOJO_PYTHON_LIBRARY="$(find $CONDA_PREFIX/lib -iname 'libpython*.[s,d]*' | sort -r | head -n 1)"
echo "export MOJO_PYTHON_LIBRARY=\"$MOJO_PYTHON_LIBRARY\"" > $CONDA_PREFIX/etc/conda/activate.d/export-mojo.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset MOJO_PYTHON_LIBRARY" > $CONDA_PREFIX/etc/conda/deactivate.d/unset-mojo.sh
```

## Usage
Activer l'environement `conda` et lancez le programme:

```bash
conda activate llama2
python3 inference.py 
# OU
mojo model.mojo
```