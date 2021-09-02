分类：从数据传输方式上来看

- 字节流：字节流读取单个字节
  - InputStream
  - OutputStream
- 字符流：字符流读取单个字符（根据编码的不同，对应的字节也不同，UTF-8是三个字节）
  - Reader
  - Writer

字节是给计算机看的，字符是给人看的。



从数据操作上

- 文件：FileInputStream、FileOutputStream、FileReader、FileWriter
- 数组：
  - 字节数组(byte[]): ByteArrayInputStream、ByteArrayOutputStream
  - 字符数组(char[]): CharArrayReader、CharArrayWriter
- 管道操作：PipedInputStream、PipedOutputStream、PipedReader、PipedWriter
- 基本数据类型：DataInputStream、DataOutputStream
- 缓冲操作：BufferedInputStream、BufferedOutputStream、BufferedReader、BufferedWriter
- 打印：PrintStream、PrintWriter
- 对象序列化和反序列化：ObjectInputStream、ObjectOutputStream
- 转换：InputStreamReader、OutputStreamWriter



### 序列化 & Serializable & transient

序列化就是将一个对象转换成字节序列，方便存储和传输。

- 序列化: ObjectOutputStream.writeObject()
- 反序列化: ObjectInputStream.readObject()

不会对静态变量进行序列化，因为序列化只是保存对象的状态，静态变量属于类的状态。

**Serializable**

序列化的类需要实现 Serializable 接口，它只是一个标准，没有任何方法需要实现，但是如果不去实现它的话而进行序列化，会抛出异常。