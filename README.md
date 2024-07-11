# TensorforceFoam

## General information
**TensorforceFoam** a TensorFlow-based intrusive Deep Reinforcement Learning (DRL) framework for **OpenFOAM**. The **Tensorforce** package is utilized for the DRL computations. The agent model is integrated within the open-source CFD solver. The integration eliminates the need for any external information exchange during DRL episodes. The framework is parallelized using the message passing interface (MPI) for Python (**mpi4py**) to manage parallel environments for computationally intensive CFD cases through distributed computing. 

The source code for the OpenFOAM implementaiton of the DRL agent is found in the `OpenFOAM` folder, while the `training` folder contains the python programs and the verification test cases. The library is tested and verified with the recent versions of [OpenFOAM](https://www.openfoam.com/), such as v2312 and 2406.

More information about the theory, the developed codes, and the test cases are presented in the published article ([An efficient intrusive deep reinforcement learning framework for OpenFOAM
](https://link.springer.com/article/10.1007/s11012-024-01830-1)).


## Installation

### Installation of the OpenFOAM side

The library is developed based on OpenFOAM ([openfoam.com](https://www.openfoam.com/)). Therefore, a complete installation of OpenFOAM (preferably a recent version, such as OpenFOAM-v2312 or v2406, is required. 

To install (compile) the semiImplicitSlip library, one needs to download source files using git:
```bash
git clone https://github.com/salehisaeed/TensorforceFoam
```
Then, in a terminal where OpenFOAM is sourced, run the `wmake` in the `OpenFOAM/src/DRLAgent` folder:
```bash
cd OpenFOAM/src/DRLAgent
wclean && wmake
```
The C++ compiler needs to be at least C++ 14 or prefereably C++ 17. The compiler settings can be set inside the following file
```bash
$WM_PROJECT_DIR/wmake/rules/General/Gcc/c++
```
Inside this file, set `-std=c++17`.


The **Tensorflow C API** needs to be installed. Download and extract the **Linux CPU only** version of TensorFlow for C. The library linked to the OpenFOAM complied library to use the Tensorflow models within C++ codes. The address for the extracted library is required in the `option` file (`OpenFOAM/src/DRLAgent/Make/option`). For convinience, an envirement variable can be defined for this address in the `.bashrc` as
```bash
export TF_LIBRARIES=/address/to/the/library/libtensorflow-cpu-linux-x86_64-2.8.0
```


The [**CppFlow**](https://github.com/serizba/cppflow) is also used here for loading and running Tensorflow models in C++. Download and install CppFlow from its corresponding github repository. However, CppFlow does not support boolean inputs to the Tensorflow models which is needed in the Tensorforce DRL agents. The boolean specifies whther the model is run in the deterministic (evaluating) or non-deterministic (for training) modes. Therefore, Small modifications are implemenetd to the original CppFlow library to make it compatibe with boolean inputs. Only two files are modified which are found in the `cppflow`folder. The rest of the cppflow files are the same as the original library.

Here again, an envirement variable can be defined for the address of the cppflow library in the `.bashrc` for convinience:
```bash
export CPPFLOW_LIBRARIES=/address/to/the/library/cppflow
```



### Installation of the Python side

Install **Tensorforce** and **mpi4py**. Look at the 





To be contibued ...!


## Reference
[1]. S. Salehi, An efficient intrusive deep reinforcement learning framework for OpenFOAM, Meccanica (2024), [doi.org/10.1007/s11012-024-01830-1](https://doi.org/10.1007/s11012-024-01830-1)
