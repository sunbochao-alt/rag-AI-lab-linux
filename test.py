import langchain
import pkgutil

print(f"📦 LangChain 安装路径: {langchain.__path__[0]}")
print("\n📂 'langchain.' 后面可以直接跟的子模块有：")
print("-" * 30)

# 扫描 langchain 文件夹下的所有子模块
for importer, modname, ispkg in pkgutil.iter_modules(langchain.__path__):
    print(f"langchain.{modname}")