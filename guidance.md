# Some explanation

## running guidance

数据集下载地址： https://drive.google.com/file/d/1KLdH0i4bIJnzconPpPXWsMcmg06PcQSH/view?usp=sharing
将数据集放在相同目录下，在相同目录下新建model_path文件夹（训练后的模型会自动保存在这里面，名字为model.path），命令行输入
'''
python train.py --mode=train
'''
来进行训练，训练完成后命令行输入
'''
python train.py --mode=test
'''
可以进行测试。

## environment

'''
packages in environment:

Name                    Version                   Build  Channel
absl-py                   2.1.0            py39haa95532_0    defaults
anyio                     4.2.0            py39haa95532_0    defaults
argon2-cffi               21.3.0             pyhd3eb1b0_0    defaults
argon2-cffi-bindings      21.2.0           py39h2bbff1b_0    defaults
asttokens                 2.0.5              pyhd3eb1b0_0    defaults
attrs                     23.1.0           py39haa95532_0    defaults
backcall                  0.2.0              pyhd3eb1b0_0    defaults
beautifulsoup4            4.12.3           py39haa95532_0    defaults
blas                      1.0                         mkl    defaults
bleach                    4.1.0              pyhd3eb1b0_0    defaults
brotli                    1.0.9                h2bbff1b_8    defaults
brotli-bin                1.0.9                h2bbff1b_8    defaults
c-ares                    1.19.1               h2bbff1b_0    defaults
ca-certificates           2024.11.26           haa95532_0    defaults
cffi                      1.16.0           py39h2bbff1b_1    defaults
click                     8.1.7            py39haa95532_0    defaults
colorama                  0.4.6            py39haa95532_0    defaults
comm                      0.2.1            py39haa95532_0    defaults
contourpy                 1.2.0            py39h59b6b97_0    defaults
cycler                    0.11.0             pyhd3eb1b0_0    defaults
debugpy                   1.6.7            py39hd77b12b_0    defaults
decorator                 5.1.1              pyhd3eb1b0_0    defaults
defusedxml                0.7.1              pyhd3eb1b0_0    defaults
dill                      0.3.8                    pypi_0    pypi
entrypoints               0.4              py39haa95532_0    defaults
exceptiongroup            1.2.0            py39haa95532_0    defaults
executing                 0.8.3              pyhd3eb1b0_0    defaults
filelock                  3.13.1                   pypi_0    pypi
fonttools                 4.51.0           py39h2bbff1b_0    defaults
freetype                  2.12.1               ha860e81_0    defaults
fsspec                    2024.2.0                 pypi_0    pypi
grpcio                    1.62.2           py39h5da7b33_0    defaults
icu                       73.1                 h6c2663c_0    defaults
idna                      3.7              py39haa95532_0    defaults
importlib-metadata        7.0.1            py39haa95532_0    defaults
importlib_resources       6.4.0            py39haa95532_0    defaults
intel-openmp              2023.1.0         h59b6b97_46320    defaults
ipykernel                 6.28.0           py39haa95532_0    defaults
ipython                   8.15.0           py39haa95532_0    defaults
ipython_genutils          0.2.0              pyhd3eb1b0_1    defaults
jedi                      0.19.1           py39haa95532_0    defaults
jinja2                    3.1.3                    pypi_0    pypi
jpeg                      9e                   h827c3e9_3    defaults
jsonschema                4.19.2           py39haa95532_0    defaults
jsonschema-specifications 2023.7.1         py39haa95532_0    defaults
jupyter_client            7.4.9            py39haa95532_0    defaults
jupyter_core              5.7.2            py39haa95532_0    defaults
jupyter_events            0.10.0           py39haa95532_0    defaults
jupyter_server            2.14.1           py39haa95532_0    defaults
jupyter_server_terminals  0.4.4            py39haa95532_1    defaults
jupyterlab_pygments       0.2.2            py39haa95532_0    defaults
kaleido                   0.1.0.post1              pypi_0    pypi
kiwisolver                1.4.4            py39hd77b12b_0    defaults
krb5                      1.20.1               h5b6d351_0    defaults
lcms2                     2.12                 h83e58a3_0    defaults
lerc                      3.0                  hd77b12b_0    defaults
libabseil                 20240116.2      cxx17_h5da7b33_0    defaults
libbrotlicommon           1.0.9                h2bbff1b_8    defaults
libbrotlidec              1.0.9                h2bbff1b_8    defaults
libbrotlienc              1.0.9                h2bbff1b_8    defaults
libclang                  14.0.6          default_hb5a9fac_1    defaults
libclang13                14.0.6          default_h8e68704_1    defaults
libdeflate                1.17                 h2bbff1b_1    defaults
libgrpc                   1.62.2               hf25190f_0    defaults
libpng                    1.6.39               h8cc25b3_0    defaults
libpq                     12.17                h906ac69_0    defaults
libprotobuf               4.25.3               hf2fb9eb_0    defaults
libsodium                 1.0.18               h62dcd97_0    defaults
libtiff                   4.5.1                hd77b12b_0    defaults
libwebp-base              1.3.2                h2bbff1b_0    defaults
lz4-c                     1.9.4                h2bbff1b_1    defaults
markdown                  3.4.1            py39haa95532_0    defaults
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.2            py39haa95532_0    defaults
matplotlib-base           3.9.2            py39he19b0ae_0    defaults
matplotlib-inline         0.1.6            py39haa95532_0    defaults
mistune                   2.0.4            py39haa95532_0    defaults
mkl                       2023.1.0         h6b88ed4_46358    defaults
mkl-service               2.4.0            py39h2bbff1b_1    defaults
mkl_fft                   1.3.10           py39h827c3e9_0    defaults
mkl_random                1.2.7            py39hc64d2fc_0    defaults
mpmath                    1.3.0                    pypi_0    pypi
nb_conda_kernels          2.3.1            py39haa95532_0    defaults
nbclassic                 1.1.0            py39haa95532_0    defaults
nbclient                  0.8.0            py39haa95532_0    defaults
nbconvert                 7.10.0           py39haa95532_0    defaults
nbformat                  5.9.2            py39haa95532_0    defaults
nest-asyncio              1.6.0            py39haa95532_0    defaults
networkx                  3.2.1                    pypi_0    pypi
notebook                  6.5.7            py39haa95532_0    defaults
notebook-shim             0.2.3            py39haa95532_0    defaults
numpy                     1.26.3                   pypi_0    pypi
numpy-base                1.26.4           py39h65a83cf_0    defaults
openjpeg                  2.5.2                hae555c5_0    defaults
openssl                   3.0.15               h827c3e9_0    defaults
overrides                 7.4.0            py39haa95532_0    defaults
packaging                 24.1             py39haa95532_0    defaults
pandocfilters             1.5.0              pyhd3eb1b0_0    defaults
parso                     0.8.3              pyhd3eb1b0_0    defaults
pickleshare               0.7.5           pyhd3eb1b0_1003    defaults
pillow                    10.2.0                   pypi_0    pypi
pip                       24.0             py39haa95532_0    defaults
platformdirs              3.10.0           py39haa95532_0    defaults
plotly                    5.24.1           py39h9909e9c_0    defaults
ply                       3.11             py39haa95532_0    defaults
prometheus_client         0.14.1           py39haa95532_0    defaults
prompt-toolkit            3.0.43           py39haa95532_0    defaults
protobuf                  4.25.3           py39h958608f_0    defaults
psutil                    5.9.0            py39h2bbff1b_0    defaults
pure_eval                 0.2.2              pyhd3eb1b0_0    defaults
py-vox-io                 0.1                      pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0    defaults
pygments                  2.15.1           py39haa95532_1    defaults
pyparsing                 3.1.2            py39haa95532_0    defaults
pyqt                      5.15.10          py39hd77b12b_0    defaults
pyqt5-sip                 12.13.0          py39h2bbff1b_0    defaults
python                    3.9.19               h1aa4202_1    defaults
python-dateutil           2.9.0post0       py39haa95532_2    defaults
python-fastjsonschema     2.16.2           py39haa95532_0    defaults
python-json-logger        2.0.7            py39haa95532_0    defaults
pywin32                   305              py39h2bbff1b_0    defaults
pywinpty                  2.0.10           py39h5da7b33_0    defaults
pyyaml                    6.0.1            py39h2bbff1b_0    defaults
pyzmq                     24.0.1           py39h2bbff1b_0    defaults
qt-main                   5.15.2              h19c9488_10    defaults
re2                       2022.04.01           hd77b12b_0    defaults
referencing               0.30.2           py39haa95532_0    defaults
rfc3339-validator         0.1.4            py39haa95532_0    defaults
rfc3986-validator         0.1.1            py39haa95532_0    defaults
rpds-py                   0.10.6           py39h062c2fa_0    defaults
scipy                     1.13.1                   pypi_0    pypi
send2trash                1.8.2            py39haa95532_0    defaults
setuptools                72.1.0           py39haa95532_0    defaults
sip                       6.7.12           py39hd77b12b_0    defaults
six                       1.16.0             pyhd3eb1b0_1    defaults
sniffio                   1.3.0            py39haa95532_0    defaults
soupsieve                 2.5              py39haa95532_0    defaults
sqlite                    3.45.3               h2bbff1b_0    defaults
stack_data                0.2.0              pyhd3eb1b0_0    defaults
sympy                     1.12                     pypi_0    pypi
tbb                       2021.8.0             h59b6b97_0    defaults
tenacity                  9.0.0            py39haa95532_0    defaults
tensorboard               2.17.0           py39haa95532_0    defaults
tensorboard-data-server   0.7.0            py39haa95532_1    defaults
terminado                 0.17.1           py39haa95532_0    defaults
tinycss2                  1.2.1            py39haa95532_0    defaults
tomli                     2.0.1            py39haa95532_0    defaults
torch                     2.4.0+cu118              pypi_0    pypi
torchaudio                2.4.0+cu118              pypi_0    pypi
torchvision               0.19.0+cu118             pypi_0    pypi
tornado                   6.4.1            py39h827c3e9_0    defaults
tqdm                      4.66.5                   pypi_0    pypi
traitlets                 5.14.3           py39haa95532_0    defaults
typing-extensions         4.9.0                    pypi_0    pypi
typing_extensions         4.11.0           py39haa95532_0    defaults
tzdata                    2024a                h04d1e81_0    defaults
unicodedata2              15.1.0           py39h2bbff1b_0    defaults
vc                        14.40                h2eaa2aa_0    defaults
vs2015_runtime            14.40.33807          h98bb1dd_0    defaults
wcwidth                   0.2.5              pyhd3eb1b0_0    defaults
webencodings              0.5.1            py39haa95532_1    defaults
websocket-client          1.8.0            py39haa95532_0    defaults
werkzeug                  3.0.6            py39haa95532_0    defaults
wheel                     0.43.0           py39haa95532_0    defaults
winpty                    0.4.3                         4    defaults
xz                        5.4.6                h8cc25b3_1    defaults
yaml                      0.2.5                he774522_0    defaults
zeromq                    4.3.5                hd77b12b_0    defaults
zipp                      3.17.0           py39haa95532_0    defaults
zlib                      1.2.13               h8cc25b3_1    defaults
zstd                      1.5.5                hd43e919_2    defaults
'''
