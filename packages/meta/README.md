# ZeroModel

`zeromodel` is the public umbrella distribution for the coordinated ZeroModel
Python runtime. Installing it installs every supported runtime component while
preserving the modular `zeromodel.*` namespace packages.

## Install

```powershell
python -m pip install zeromodel
```

The umbrella owns no implementation namespace. Runtime code is provided by
component distributions such as `zeromodel-core`, `zeromodel-analysis`,
`zeromodel-vision`, and `zeromodel-search`.

Advanced users can still install individual components:

```powershell
python -m pip install zeromodel-core
python -m pip install zeromodel-vision
python -m pip install zeromodel-search
```
