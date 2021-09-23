- conda --version #查看conda版本，验证是否安装

- conda update conda #更新至最新版本，也会更新其它相关包

- conda create -n package_name #创建名为env_name的新环境，并在该环境下安装名为package_name 的包，可以指定新环境的版本号，

  例如：conda create -n python2 python=python2.7 numpy pandas，创建了python2环境，python版本为2.7，同时还安装了numpy pandas包

- source activate env_name #切换至env_name环境

- source deactivate #退出环境

- conda info -e #显示所有已经创建的环境

- conda remove --name env_name –all #删除环境

- conda list #查看所有已经安装的包

- conda install matplotlib 安装库

  







不常用：

- `conda config --set auto_activate_base false：可以通过配置`auto_activate_base`关闭自动进入conda基础环境：`

换源：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes 