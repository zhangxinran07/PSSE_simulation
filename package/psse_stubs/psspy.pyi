# psspy 类型桩(PSSE 只提供 psspy.pyc, Pylance 无法解析字节码)
# 声明为动态属性模块: 消除 "无法解析导入" 警告, 属性访问按 Any 处理(不做类型检查)
from typing import Any
def __getattr__(name: str) -> Any: ...
