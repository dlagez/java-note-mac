## 二叉树

定义一个二叉树

```java
public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
      TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
  }
```

二叉树的遍历：

```java 
/* 二叉树遍历框架 */
void traverse(TreeNode root) {
    // 前序遍历 中左右
    traverse(root.left)
    // 中序遍历 左中右
    traverse(root.right)
    // 后序遍历 左右中
}
```

打印目录下所有的文件（包括目录的目录里的文件）

```java
    public static void main(String[] args) {
        File file = new File("/Volumes/roczhang/book");
        File[] files1 = file.listFiles();  // 列出所有文件和子目录
        printFiles(files1);
        printFiles(file);
    }

static void printFiles(File file) {
    File[] files = file.listFiles();
    for (File f : files) {
        if (f.isDirectory()){
            System.out.println("======begin======="+f+"========begin========");
            printFiles(f);  // 若是目录，递归打印该目录下的文件
            System.out.println("======end======="+f+"========end========");
        }
        if (f.isFile())
            System.out.println(f);  // 若是文件，直接打印
    }
}
```

复制文件

```java
public class demo5_copyFile {
    public static void main(String[] args) throws IOException {
        File input = new File("/Volumes/roczhang/temp/temp.txt");
        File output = new File("/Volumes/roczhang/temp/temp2.txt");
        copyFile(input, output);
    }

    public static void copyFile(File source, File target) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(source); FileOutputStream outputStream = new FileOutputStream(target)) {
            byte[] bytes = new byte[1000];
            int byteRead;
            while ((byteRead = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, byteRead);
            }
        }
    }
}
```

