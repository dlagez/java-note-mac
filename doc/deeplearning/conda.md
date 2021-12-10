miniconda官网：[Miniconda — Conda documentation](https://docs.conda.io/en/latest/miniconda.html#)

### 换源

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 创建环境

```
conda create -n bigdata python=3.8
```

### 临时使用pip源

```
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装jupyter

```
conda install jupyter notebook
```

### 修改根目录

生成配置文件 

```
jupyter notebook --generate-config
```

修改配置文件jupyter_notebook_config.py中的 c.NotebookApp.notebook_dir = ‘’ 改为要修改的根目录把单引号换成双引号  c.NotebookApp.notebook_dir = "C:/roczhang"

### 将环境添加到kernel，

```
pip install ipykernel
python -m ipykernel install --name bigdata --display-name "bigdata"
```

- `--display-name`指定jupyter notebook中显示的名字

升级包

```
onda update package
```

