Control Room 

## 发布Process

Process Studio  仅仅用于创建和测试Process。但是再真实的环境中，Process不在Process Studio 模块运行。它再Control Room里面运行。

为了防止有破坏性的Process直接运行，所以我们创建了Process之后需要发布才能再Controller Room里面看到。

第一步：点击导航栏的Studio，然后可以看到一下信息，为了能再Control Room里面找到你开发完成的Process。你需要进入到Process里面。这里我们以demo2为例子。双击进入流程。

![image-20220106193310389](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106193310389.png)

第二步：右击红色方框包裹的流程。

![image-20220106193623767](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106193623767.png)

第三步：选择他的属性。

![image-20220106193802076](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106193802076.png)

第四步：勾选Publish，点击OK

![image-20220106194140422](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106194140422.png)

第五步：记得保存你做的修改。

![image-20220106194035264](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106194035264.png)

大功告成！

![image-20220106194319851](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106194319851.png)



将Process添加到环境中：

在Environment中可以看到demo1的status是pending，这个状态表示这个Process已经准备好了，可以start了。

![image-20220106194914592](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106194914592.png)



运行Process

![image-20220106195607590](https://cdn.jsdelivr.net/gh/dlagez/img@master/image-20220106195607590.png)