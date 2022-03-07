mac使用visdom的大坑：

安装很简单：

```bash
pip install visdom
```

但是使用的时候：

```
python -m visdom.server
```

会弹出下面的提示：

这里解释一下：`reset by peer`表示目标服务器拒绝了你的下载请求，由于下载的文件在国外。你懂的。

```
(pytorch) roczhang@roczhang-mac DeepHyperX % python -m visdom.server
/Users/roczhang/miniforge3/envs/pytorch/lib/python3.8/site-packages/visdom/server.py:39: DeprecationWarning: zmq.eventloop.ioloop is deprecated in pyzmq 17. pyzmq now works with default tornado and asyncio eventloops.
  ioloop.install()  # Needs to happen before any tornado imports!
Checking for scripts.
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/react-modal@3.1.10/dist/react-modal.min.js
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/classnames@2.2.5
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.woff2
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.woff
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.ttf
ERROR:root:Error [Errno 54] Connection reset by peer while downloading https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular
It's Alive!
INFO:root:Application Started
You can navigate to http://localhost:8097
```

打开这个地址`http://localhost:8097`会直接蓝屏。



查看报错并看看能不能解决它：这个文件报的错，点击它即可查看该文件。

```
/Users/roczhang/miniforge3/envs/pytorch/lib/python3.8/site-packages/visdom/server.py
```

![image-20220306214110942](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220306214110942.png)

我看了整个文件发现在它的最后面有下载代码：下载代码和报错代码一致，指定就是这里除了问题。

![Untitled7](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled7.png)

把代码抠出来细看：发现这个小东西下载文件的时候还把原始文件名给改了，不地道。由于我们不知道需要下载哪些文件，所以我们先打开这个地址`http://localhost:8097`看看控制台会报什么错误，我们就知道需要下载什么文件了。

```python
def download_scripts(proxies=None, install_dir=None):
    import visdom
    print("Checking for scripts.")

    # location in which to download stuff:
    if install_dir is None:
        install_dir = os.path.dirname(visdom.__file__)

    # all files that need to be downloaded:
    b = 'https://unpkg.com/'
    bb = '%sbootstrap@3.3.7/dist/' % b
    ext_files = {
        # - js
        '%sjquery@3.1.1/dist/jquery.min.js' % b: 'jquery.min.js',
        '%sbootstrap@3.3.7/dist/js/bootstrap.min.js' % b: 'bootstrap.min.js',
        '%sreact@16.2.0/umd/react.production.min.js' % b: 'react-react.min.js',
        '%sreact-dom@16.2.0/umd/react-dom.production.min.js' % b:
            'react-dom.min.js',
        '%sreact-modal@3.1.10/dist/react-modal.min.js' % b:
            'react-modal.min.js',
        'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_SVG':  # noqa
            'mathjax-MathJax.js',
        # here is another url in case the cdn breaks down again.
        # https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly.min.js
        'https://cdn.plot.ly/plotly-latest.min.js': 'plotly-plotly.min.js',
        # Stanford Javascript Crypto Library for Password Hashing
        '%ssjcl@1.0.7/sjcl.js' % b: 'sjcl.js',

        # - css
        '%sreact-resizable@1.4.6/css/styles.css' % b:
            'react-resizable-styles.css',
        '%sreact-grid-layout@0.16.3/css/styles.css' % b:
            'react-grid-layout-styles.css',
        '%scss/bootstrap.min.css' % bb: 'bootstrap.min.css',

        # - fonts
        '%sclassnames@2.2.5' % b: 'classnames',
        '%slayout-bin-packer@1.4.0/dist/layout-bin-packer.js' % b:
            'layout_bin_packer.js',
        '%sfonts/glyphicons-halflings-regular.eot' % bb:
            'glyphicons-halflings-regular.eot',
        '%sfonts/glyphicons-halflings-regular.woff2' % bb:
            'glyphicons-halflings-regular.woff2',
        '%sfonts/glyphicons-halflings-regular.woff' % bb:
            'glyphicons-halflings-regular.woff',
        '%sfonts/glyphicons-halflings-regular.ttf' % bb:
            'glyphicons-halflings-regular.ttf',
        '%sfonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular' % bb:  # noqa
            'glyphicons-halflings-regular.svg#glyphicons_halflingsregular',
    }

    # make sure all relevant folders exist:
    dir_list = [
        '%s' % install_dir,
        '%s/static' % install_dir,
        '%s/static/js' % install_dir,
        '%s/static/css' % install_dir,
        '%s/static/fonts' % install_dir,
    ]
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # set up proxy handler:
    from six.moves.urllib import request
    from six.moves.urllib.error import HTTPError, URLError
    handler = request.ProxyHandler(proxies) if proxies is not None \
        else request.BaseHandler()
    opener = request.build_opener(handler)
    request.install_opener(opener)

    built_path = os.path.join(here, 'static/version.built')
    is_built = visdom.__version__ == 'no_version_file'
    if os.path.exists(built_path):
        with open(built_path, 'r') as build_file:
            build_version = build_file.read().strip()
        if build_version == visdom.__version__:
            is_built = True
        else:
            os.remove(built_path)
    if not is_built:
        print('Downloading scripts, this may take a little while')

    # download files one-by-one:
    for (key, val) in ext_files.items():

        # set subdirectory:
        if val.endswith('.js'):
            sub_dir = 'js'
        elif val.endswith('.css'):
            sub_dir = 'css'
        else:
            sub_dir = 'fonts'

        # download file:
        filename = '%s/static/%s/%s' % (install_dir, sub_dir, val)
        if not os.path.exists(filename) or not is_built:
            req = request.Request(key,
                                  headers={'User-Agent': 'Chrome/30.0.0.0'})
            try:
                data = opener.open(req).read()
                with open(filename, 'wb') as fwrite:
                    fwrite.write(data)
            except HTTPError as exc:
                logging.error('Error {} while downloading {}'.format(
                    exc.code, key))
            except URLError as exc:
                logging.error('Error {} while downloading {}'.format(
                    exc.reason, key))

    if not is_built:
        with open(built_path, 'w+') as build_file:
            build_file.write(visdom.__version__)
```



![Untitled8](https://cdn.jsdelivr.net/gh/dlagez/img@master/Untitled8.png)

这里我随便找一个文件来下载：http://localhost:8097/static/css/bootstrap.min.css.map