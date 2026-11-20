ExoCLASS+DarkHistory (ExoCLASS-DH)  {#mainpage}
========

ExoCLASS-DH is a version of the Boltzmann code [CLASS](https://github.com/lesgourg/class_public)
extended to handle exotic energy injection (e.g. dark matter annihilation and
decay). It computes energy deposition with the DarkAges module, and it can be
coupled to [DarkHistory](https://github.com/hongwanliu/DarkHistory), which is
included as a git submodule.

Installing ExoCLASS-DH
----------------------

### 1. Clone the repository with its submodule

<!-- TODO: update this URL to the final public location of the repository. -->

```bash
git clone --recurse-submodules https://github.com/PoulinV/ExoCLASS_perso
cd ExoCLASS_perso
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

This checks out DarkHistory (branch `CLASS_interface`, at the commit pinned in
this repository) into `DH_interface/DarkHistory/`. Leave it at that path:
CLASS calls `DH_interface/DarkHistory/DHoneline.py` from there.

### 2. Set up a Python environment (Python >= 3.10)

ExoCLASS runs DarkHistory and DarkAges with a plain `python` command, so
`python` (not only `python3`) must be on your `PATH`, and must have the
packages below, in the shell where you run `./class`. Create and activate a
conda environment with all the dependencies:

```bash
conda create -n exoclass -c conda-forge python=3.12 numpy scipy cython astropy h5py matplotlib tqdm pyyaml dill future
conda activate exoclass
```

DarkHistory itself does not need to be installed: CLASS runs it directly from
the submodule, so Python finds the `darkhistory` package there.

*Optional:* to `import darkhistory` in your own scripts or notebooks, also
install it into the environment. Use `--no-deps` so that pip does not touch
the packages installed by conda:

```bash
pip install --no-deps -e DH_interface/DarkHistory
```

### 3. Download the DarkHistory data files

DarkHistory needs data tables that are not stored in git. Download them 
at https://doi.org/10.5281/zenodo.13259508.

Point DarkHistory to this directory, for example in your
`~/.zshrc` or `~/.bashrc`:

```bash
export DH_DATA_DIR=/path/to/DarkHistory_data
```

These files are only needed for the full DarkHistory interface (see below).

### 4. Compile ExoCLASS and its Python wrapper

From the repository root:

```bash
make clean
make -j          # builds ./class, libclass.a and the classy Python wrapper
```

The absolute path of the repository is compiled into the code. If you move
the repository, run `make clean; make` again.

### 5. Check the installation

```bash
./class explanatory.ini                  # plain CLASS
./class example_ExoCLASS-DH-fast.ini     # fast interface
./class example_ExoCLASS-DH-full.ini     # full interface (runs DarkHistory; slow)
python -c "import classy; print('ok')"
```

Fast vs. full DarkHistory interface
-----------------------------------

- **Fast interface** (`example_ExoCLASS-DH-fast.ini`): uses precomputed
  transfer tables, committed under `DarkAgesModule/transfer_functions/`,
  through the DarkAges module. It does not run DarkHistory and does not need
  the DarkHistory data files.
- **Full interface** (`example_ExoCLASS-DH-full.ini`): calls DarkHistory
  during the run. It needs the submodule, the Python packages and the data
  files from steps 1–3, and is much slower.

The example `.ini` files explain the available options.

Citing ExoCLASS
---------------

<!-- TODO: update these references when papers are out. -->
If you use ExoCLASS, please cite
[arXiv:1801.01871](https://arxiv.org/abs/1801.01871) and
[arXiv:1910.04619](https://arxiv.org/abs/1910.04619), as well as the CLASS
paper below. If you use the DarkHistory interface, please also cite
DarkHistory: [arXiv:1904.09296](https://arxiv.org/abs/1904.09296),
[arXiv:2303.07366](https://arxiv.org/abs/2303.07366) and
[arXiv:2303.07370](https://arxiv.org/abs/2303.07370).


Original CLASS documentation
============================

CLASS: Cosmic Linear Anisotropy Solving System

Authors: Julien Lesgourgues, Thomas Tram, Nils Schoeneberg

with several major inputs from other people, especially Benjamin
Audren, Simon Prunet, Jesus Torrado, Miguel Zumalacarregui, Francesco
Montanari, Deanna Hooper, Samuel Brieden, Daniel Meinert, Matteo Lucca, etc.

For information on CLASS, see http://class-code.net


Compiling CLASS and getting started
-----------------------------------

To compile only the C code, go to the repository root and run (make clean;
make class). You can usually speed up compilation with the option -j:
make -j class. If the first compilation attempt fails, you may need to
open the Makefile and adapt the name of the compiler (default: gcc),
of the optimization flag (default: -O4 -ffast-math) and of the OpenMP
flag (default: -fopenmp; this flag is facultative, you are free to
compile without OpenMP if you don't want parallel execution; note that
you need the version 4.2 or higher of gcc to be able to compile with
-fopenmp). Many more details on the CLASS compilation are given on the
wiki page

https://github.com/lesgourg/class_public/wiki/Installation

(in particular, for compiling on Mac >= 10.9 despite of the clang
incompatibility with OpenMP).

To check that the code runs, type:

    ./class explanatory.ini

The explanatory.ini file is THE reference input file, containing and
explaining the use of all possible input parameters. We recommend to
read it, to keep it unchanged (for future reference), and to create
for your own purposes some shorter input files, containing only the
input lines which are useful for you. Input files must have a *.ini
extension. We provide an example of an input file containing a
selection of the most used parameters, default.ini, that you may use as a
starting point.

If you want to play with the precision/speed of the code, you can use
one of the provided precision files (e.g. cl_permille.pre) or modify
one of them, and run with two input files, for instance:

    ./class test.ini cl_permille.pre

The files *.pre are suppposed to specify the precision parameters for
which you don't want to keep default values. If you find it more
convenient, you can pass these precision parameter values in your *.ini
file instead of an additional *.pre file.

The automatically-generated documentation is located in

    doc/manual/html/index.html
    doc/manual/CLASS_manual.pdf

On top of that, if you wish to modify the code, you will find lots of
comments directly in the files.

Python
------

To use CLASS from python, or ipython notebooks, or from the Monte
Python parameter extraction code, you need to compile not only the
code, but also its python wrapper. This can be done by typing just
'make' instead of 'make class' (or for speeding up: 'make -j'). More
details on the wrapper and its compilation are found on the wiki page

https://github.com/lesgourg/class_public/wiki

Plotting utility
----------------

Since version 2.3, the package includes an improved plotting script
called CPU.py (Class Plotting Utility), written by Benjamin Audren and
Jesus Torrado. It can plot the Cl's, the P(k) or any other CLASS
output, for one or several models, as well as their ratio or percentage
difference. The syntax and list of available options is obtained by
typing 'pyhton CPU.py -h'. There is a similar script for MATLAB,
written by Thomas Tram. To use it, once in MATLAB, type 'help
plot_CLASS_output.m'

Developing the code
--------------------

If you want to develop the code, we suggest that you download it from
the github webpage

https://github.com/lesgourg/class_public

rather than from class-code.net. Then you will enjoy all the feature
of git repositories. You can even develop your own branch and get it
merged to the public distribution. For related instructions, check

https://github.com/lesgourg/class_public/wiki/Public-Contributing

Using the code
--------------

You can use CLASS freely, provided that in your publications, you cite
at least the paper `CLASS II: Approximation schemes <http://arxiv.org/abs/1104.2933>`. Feel free to cite more CLASS papers!

Support
-------

To get support, please open a new issue on the

https://github.com/lesgourg/class_public

webpage!
