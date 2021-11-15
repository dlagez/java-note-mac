### 后台代码

```java
@GetMapping("download")
    public Object downloadFile(HttpServletRequest req, HttpServletResponse resp) {
        OutputStream os = null;
        InputStream in = null;
        try {
            os = resp.getOutputStream();
            resp.reset();
            resp.setContentType("application/x-download;charset=GBK");
            resp.setHeader("Content-Disposition", "attachment");
            // 读取本地文件
            File f = new File("C:\\roczhang\\tmp\\tmp.jpg");
            in = new FileInputStream(f);
            if (in == null) {
                return "下载附件失败，请检查文件是否存在";
            }
            // 这里可以换成直接将服务器的图片复制到outputstream
            IOUtils.copy(in, resp.getOutputStream());
            resp.getOutputStream().flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            // 关闭输入流
            try {
                if (in != null) {
                    in.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            // 关闭输出流
            try {
                if (os != null) {
                    os.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
```

前端代码

```html
<a href="http://localhost:8081/download">点击下载图片</a>
```

