# MBNPy toolkit

## Description
MBNPy is a Bayesian Network toolkit designed to handle a large number of parent nodes—problems where conventional BN tools often fall short.  
Example applications include [transport networks](https://doi.org/10.1016/j.ress.2019.01.007) and [pipeline networks](https://doi.org/10.1016/j.ress.2021.107468).

## Contact
If you have discussion points, refer to the [discussions tab](https://github.com/jieunbyun/BNS-JT/discussions).  
If you have need support, refer to the [issues tab](https://github.com/jieunbyun/BNS-JT/issues).

## Installation
```
# Downloading files
git clone git@github.com:jieunbyun/BNS-JT.git
cd <BNS-JT dir>

# Creating environment using venv
python3 -m venv <venv dir>
source <venv dir>/bin/activate
pip install -r requirements_py3.9.txt

# Or creating environment using conda
conda env create --name bns --file BNS_JT_py3.9.yml
conda activate bns
```

## Documentation
Coming soon.

## License
MBNPy is distributed under the MIT License

Copyright (c) <2025> <Ji-Eun Byun>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Referencing MBNPy
If you are using this software for publication, please cite this paper:

Byun, J. E. & Song, J. (2021). Generalized matrix-based Bayesian network for multi-state systems. *Reliability Engineering & System Safety,* 211, 107468.