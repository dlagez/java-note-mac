BeanUtils位于spring-beans模块中。静态的copyProperties方法进行属性的copy。

```java
private static void copyProperties(Object source, Object target, @Nullable Class<?> editable, @Nullable String... ignoreProperties) throws BeansException {
    Assert.notNull(source, "Source must not be null");
    Assert.notNull(target, "Target must not be null");
    Class<?> actualEditable = target.getClass();
    if (editable != null) {
        if (!editable.isInstance(target)) {
            throw new IllegalArgumentException("Target class [" + target.getClass().getName() + "] not assignable to Editable class [" + editable.getName() + "]");
        }

        actualEditable = editable;
    }
    // 调用java的api 获取 PropertyDescriptor 这个是目标的
    PropertyDescriptor[] targetPds = getPropertyDescriptors(actualEditable);
    List<String> ignoreList = ignoreProperties != null ? Arrays.asList(ignoreProperties) : null;
    PropertyDescriptor[] var7 = targetPds;
    int var8 = targetPds.length;
    
    // 遍历目标bean的PropertyDescriptor
    for(int var9 = 0; var9 < var8; ++var9) {
        PropertyDescriptor targetPd = var7[var9];
        Method writeMethod = targetPd.getWriteMethod();
        // 判断是否存在setter方法以及属性是否在需要忽略的列表中
        if (writeMethod != null && (ignoreList == null || !ignoreList.contains(targetPd.getName()))) {
           //  获取源的PropertyDescriptor
           PropertyDescriptor sourcePd = getPropertyDescriptor(source.getClass(), targetPd.getName());
            if (sourcePd != null) {
                // 获取getter方法
                Method readMethod = sourcePd.getReadMethod();
                if (readMethod != null) {
                  	// 源是可读的，目标是可写的
                    ResolvableType sourceResolvableType = ResolvableType.forMethodReturnType(readMethod);
                    ResolvableType targetResolvableType = ResolvableType.forMethodParameter(writeMethod, 0);
                    boolean isAssignable = !sourceResolvableType.hasUnresolvableGenerics() && !targetResolvableType.hasUnresolvableGenerics() ? targetResolvableType.isAssignableFrom(sourceResolvableType) : ClassUtils.isAssignable(writeMethod.getParameterTypes()[0], readMethod.getReturnType());
                  // 上面两个要求都满足了  
                  if (isAssignable) {
                        try {
                          	// 如果getter方法不是public，则需要设置其可获取
                            if (!Modifier.isPublic(readMethod.getDeclaringClass().getModifiers())) {
                                readMethod.setAccessible(true);
                            }
	                          // 反射获取属性值
                            Object value = readMethod.invoke(source);
                          // 如果setter方法不是public，则需要设置其可获取  
                          if (!Modifier.isPublic(writeMethod.getDeclaringClass().getModifiers())) {
                                writeMethod.setAccessible(true);
                            }
														// 反射赋值。
                            writeMethod.invoke(target, value);
                        } catch (Throwable var18) {
                            throw new FatalBeanException("Could not copy property '" + targetPd.getName() + "' from source to target", var18);
                        }
                    }
                }
            }
        }
    }

}
```